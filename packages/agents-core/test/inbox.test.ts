import { beforeAll, describe, expect, it } from 'vitest';
import {
  Agent,
  AgentInputItem,
  InputGuardrailTripwireTriggered,
  MaxTurnsExceededError,
  MemorySession,
  RunInbox,
  RunUserInputItem,
  Runner,
  RunState,
  setDefaultModelProvider,
  setTracingDisabled,
  user,
  withTrace,
} from '../src';
import { drainInboxIntoState } from '../src/runner/inbox';
import { RunContext } from '../src/runContext';
import {
  FakeModel,
  FakeModelProvider,
  fakeModelMessage,
  TEST_MODEL_FUNCTION_CALL,
  TEST_MODEL_MESSAGE,
  TEST_MODEL_RESPONSE_BASIC,
} from './stubs';
import { Model, ModelRequest, ModelResponse } from '../src/model';
import * as protocol from '../src/types/protocol';
import { Usage } from '../src/usage';
import { tool } from '../src/tool';
import { z } from 'zod';

function getFirstTextContent(item: AgentInputItem): string | undefined {
  if (item.type !== 'message') {
    return undefined;
  }
  if (typeof item.content === 'string') {
    return item.content;
  }
  if (Array.isArray(item.content)) {
    const first = item.content[0] as { text?: string };
    return first?.text;
  }
  return undefined;
}

function getRequestInputItems(request: ModelRequest): AgentInputItem[] {
  return Array.isArray(request.input) ? request.input : [];
}

/**
 * A model that records the input items each `getResponse` call sees and returns the
 * pre-configured response for that call. Lets tests assert exactly what the model saw on
 * each turn.
 */
class RecordingModel implements Model {
  public readonly seenInputs: AgentInputItem[][] = [];

  constructor(
    private readonly responses: ModelResponse[],
    private readonly hooks: { onCall?: (callIndex: number) => void } = {},
  ) {}

  async getResponse(req: ModelRequest): Promise<ModelResponse> {
    const callIndex = this.seenInputs.length;
    this.seenInputs.push(getRequestInputItems(req));
    this.hooks.onCall?.(callIndex);
    const response = this.responses[callIndex];
    if (!response) {
      throw new Error(`No response configured for call #${callIndex}`);
    }
    return response;
  }

  async *getStreamedResponse(
    req: ModelRequest,
  ): AsyncIterable<protocol.StreamEvent> {
    const response = await this.getResponse(req);
    yield {
      type: 'response_done',
      response: {
        id: `r_${this.seenInputs.length}`,
        usage: {
          requests: 1,
          inputTokens: response.usage.inputTokens,
          outputTokens: response.usage.outputTokens,
          totalTokens: response.usage.totalTokens,
        },
        output: response.output,
      },
    } as any;
  }
}

function functionCall(
  callId: string,
  name = 'noop',
): protocol.FunctionCallItem {
  return {
    type: 'function_call',
    id: `fc_${callId}`,
    callId,
    name,
    status: 'completed',
    arguments: '{}',
  } as protocol.FunctionCallItem;
}

describe('RunInbox', () => {
  beforeAll(() => {
    setTracingDisabled(true);
  });

  describe('queue primitives', () => {
    it('coerces string sends to user messages', () => {
      const inbox = new RunInbox();
      inbox.send('hello there');
      expect(inbox.size).toBe(1);
      const drained = inbox.drain();
      expect(drained).toHaveLength(1);
      expect(drained[0].role).toBe('user');
      expect(getFirstTextContent(drained[0])).toBe('hello there');
    });

    it('accepts pre-built UserMessageItems and arrays', () => {
      const inbox = new RunInbox();
      inbox.send(user('one'));
      inbox.send([user('two'), 'three']);
      expect(inbox.size).toBe(3);
      const drained = inbox.drain();
      expect(drained.map((m) => getFirstTextContent(m))).toEqual([
        'one',
        'two',
        'three',
      ]);
    });

    it('drain leaves the inbox empty and is idempotent on empty', () => {
      const inbox = new RunInbox();
      inbox.send('a');
      expect(inbox.drain()).toHaveLength(1);
      expect(inbox.size).toBe(0);
      expect(inbox.drain()).toEqual([]);
    });
  });

  describe('drainInboxIntoState helper', () => {
    it('returns empty when inbox is undefined or empty', async () => {
      const agent = new Agent({ name: 'A', model: new FakeModel() });
      const state = new RunState(new RunContext(), 'hi', agent, 5);
      expect(await drainInboxIntoState(state, undefined, [])).toEqual([]);
      const inbox = new RunInbox();
      expect(await drainInboxIntoState(state, inbox, [])).toEqual([]);
      expect(state._generatedItems).toHaveLength(0);
    });

    it('appends RunUserInputItems to _generatedItems in order', async () => {
      const agent = new Agent({ name: 'A', model: new FakeModel() });
      const state = new RunState(new RunContext(), 'hi', agent, 5);
      const inbox = new RunInbox();
      inbox.send(['follow up one', 'follow up two']);
      const drained = await drainInboxIntoState(state, inbox, []);
      expect(drained).toHaveLength(2);
      expect(state._generatedItems).toHaveLength(2);
      expect(state._generatedItems[0]).toBeInstanceOf(RunUserInputItem);
      expect((state._generatedItems[0] as RunUserInputItem).content).toBe(
        'follow up one',
      );
      expect((state._generatedItems[1] as RunUserInputItem).content).toBe(
        'follow up two',
      );
    });

    it('runs configured input guardrails on the drained batch and throws on tripwire', async () => {
      const agent = new Agent({
        name: 'GuardedAgent',
        model: new FakeModel(),
        inputGuardrails: [
          {
            name: 'banned-word',
            execute: async ({ input }) => {
              const text = JSON.stringify(input);
              return {
                tripwireTriggered: text.includes('forbidden'),
                outputInfo: { reason: 'banned' },
              };
            },
          },
        ],
      });
      const state = new RunState(new RunContext(), 'hi', agent, 5);
      const inbox = new RunInbox();
      inbox.send('this is forbidden content');
      await expect(
        withTrace('inbox-guardrail-trip', () =>
          drainInboxIntoState(state, inbox, []),
        ),
      ).rejects.toThrow(/Input guardrail triggered/);
      // Tripped batch is dropped, not partially persisted.
      expect(state._generatedItems).toHaveLength(0);
    });
  });
});

describe('Runner.run with inbox (non-streaming)', () => {
  beforeAll(() => {
    setTracingDisabled(true);
    setDefaultModelProvider(new FakeModelProvider());
  });

  it('delivers a queued message before the next turn and includes it in result.history', async () => {
    // Turn 1: model calls a tool. Turn 2: model produces a final message.
    // The inbox message is queued during turn 1's tool execution, so turn 2's input must
    // include it positioned chronologically after turn 1's items.
    const responses: ModelResponse[] = [
      {
        output: [functionCall('call_1', 'noop')],
        usage: new Usage(),
      },
      {
        output: [{ ...TEST_MODEL_MESSAGE }],
        usage: new Usage(),
      },
    ];

    const inbox = new RunInbox();
    const model = new RecordingModel(responses, {
      onCall: (callIndex) => {
        // After the first model call returns a tool call, queue an inbox message before the
        // tool actually executes. The drain at the top of the next turn should pick it up.
        if (callIndex === 0) {
          inbox.send('please also include the weather');
        }
      },
    });

    const agent = new Agent({
      name: 'InboxAgent',
      model,
      tools: [
        {
          type: 'function',
          name: 'noop',
          description: 'no op',
          strict: true,
          parameters: {
            type: 'object',
            properties: {},
            additionalProperties: false,
          },
          needsApproval: async () => false,
          invoke: async () => 'ok',
        } as any,
      ],
    });

    const result = await new Runner().run(agent, 'kick off the run', { inbox });

    expect(model.seenInputs).toHaveLength(2);
    const turn2Inputs = model.seenInputs[1];
    const userTexts = turn2Inputs
      .filter(
        (item) => item.type === 'message' && (item as any).role === 'user',
      )
      .map((item) => getFirstTextContent(item));
    expect(userTexts).toContain('kick off the run');
    expect(userTexts).toContain('please also include the weather');

    // Inbox message must appear in history AFTER the function call and tool output, not before.
    const history = result.history;
    const callIdx = history.findIndex((item) => item.type === 'function_call');
    const inboxIdx = history.findIndex(
      (item) =>
        item.type === 'message' &&
        (item as any).role === 'user' &&
        getFirstTextContent(item) === 'please also include the weather',
    );
    expect(callIdx).toBeGreaterThanOrEqual(0);
    expect(inboxIdx).toBeGreaterThan(callIdx);

    // Inbox is drained after delivery.
    expect(inbox.size).toBe(0);
  });

  it('preserves drained inbox messages on `state` when the next turn exceeds maxTurns', async () => {
    // Regression guard: drainInboxIntoState pushes RunUserInputItems onto state._generatedItems
    // BEFORE prepareTurn enforces maxTurns, and MaxTurnsExceededError carries `state`. So even
    // when the very next turn would max out, the drained content is reachable via `error.state`
    // and the caller can resume.
    const responses: ModelResponse[] = [
      {
        output: [functionCall('call_1', 'noop')],
        usage: new Usage(),
      },
    ];

    const inbox = new RunInbox();
    const model = new RecordingModel(responses, {
      onCall: (callIndex) => {
        if (callIndex === 0) {
          inbox.send('queued mid-run before maxTurns trips');
        }
      },
    });

    const agent = new Agent({
      name: 'InboxAgent',
      model,
      tools: [
        {
          type: 'function',
          name: 'noop',
          description: 'no op',
          strict: true,
          parameters: {
            type: 'object',
            properties: {},
            additionalProperties: false,
          },
          needsApproval: async () => false,
          invoke: async () => 'ok',
        } as any,
      ],
    });

    let caught: unknown;
    try {
      await new Runner().run(agent, 'kick off', { inbox, maxTurns: 1 });
    } catch (error) {
      caught = error;
    }

    expect(caught).toBeInstanceOf(MaxTurnsExceededError);
    const error = caught as MaxTurnsExceededError;
    expect(error.state).toBeDefined();
    const inboxItems = error.state!._generatedItems.filter(
      (item) => item instanceof RunUserInputItem,
    ) as RunUserInputItem[];
    expect(inboxItems).toHaveLength(1);
    expect(inboxItems[0].content).toBe('queued mid-run before maxTurns trips');
    expect(inbox.size).toBe(0);
  });

  it('extends the loop by one turn when a message arrives during the final model call', async () => {
    // Turn 1: model returns a final message. Without the inbox feature this would terminate.
    // Turn 2: must run because a message was queued during turn 1.
    const responses: ModelResponse[] = [
      {
        output: [fakeModelMessage('first answer')],
        usage: new Usage(),
      },
      {
        output: [fakeModelMessage('second answer with new context')],
        usage: new Usage(),
      },
    ];

    const inbox = new RunInbox();
    const model = new RecordingModel(responses, {
      onCall: (callIndex) => {
        if (callIndex === 0) {
          inbox.send('actually here is more info');
        }
      },
    });

    const agent = new Agent({ name: 'ExtendAgent', model });
    const result = await new Runner().run(agent, 'do the thing', { inbox });

    expect(model.seenInputs).toHaveLength(2);
    expect(result.finalOutput).toBe('second answer with new context');
    expect(inbox.size).toBe(0);

    const turn2UserTexts = model.seenInputs[1]
      .filter(
        (item) => item.type === 'message' && (item as any).role === 'user',
      )
      .map((item) => getFirstTextContent(item));
    expect(turn2UserTexts).toContain('actually here is more info');
  });

  it('does not extend the loop when the inbox is empty at final_output', async () => {
    const model = new RecordingModel([
      {
        output: [{ ...TEST_MODEL_MESSAGE }],
        usage: new Usage(),
      },
    ]);
    const inbox = new RunInbox();
    const agent = new Agent({ name: 'NoExtendAgent', model });
    const result = await new Runner().run(agent, 'hello', { inbox });
    expect(model.seenInputs).toHaveLength(1);
    expect(result.finalOutput).toBeDefined();
  });

  it('runs unchanged when no inbox is provided', async () => {
    const model = new RecordingModel([
      {
        output: [{ ...TEST_MODEL_MESSAGE }],
        usage: new Usage(),
      },
    ]);
    const agent = new Agent({ name: 'NoInboxAgent', model });
    const result = await new Runner().run(agent, 'hello');
    expect(result.finalOutput).toBeDefined();
    expect(model.seenInputs).toHaveLength(1);
  });
});

describe('RunInbox + RunState serialization', () => {
  beforeAll(() => {
    setTracingDisabled(true);
  });

  it('serializes user_input_item entries and round-trips them through fromString', async () => {
    const agent = new Agent({ name: 'SerializeAgent', model: new FakeModel() });
    const state = new RunState(new RunContext(), 'first prompt', agent, 5);
    const inbox = new RunInbox();
    inbox.send('queued before serialize');
    await drainInboxIntoState(state, inbox, []);

    const serialized = state.toString();
    const restored = await RunState.fromString(agent, serialized);

    const restoredItems = restored._generatedItems.filter(
      (item): item is RunUserInputItem => item instanceof RunUserInputItem,
    );
    expect(restoredItems).toHaveLength(1);
    expect(restoredItems[0].content).toBe('queued before serialize');
  });

  it('refuses to load older schema versions that contain user_input_item', async () => {
    const agent = new Agent({ name: 'SerializeAgent', model: new FakeModel() });
    const state = new RunState(new RunContext(), 'first prompt', agent, 5);
    const inbox = new RunInbox();
    inbox.send('queued');
    await drainInboxIntoState(state, inbox, []);

    const json = JSON.parse(state.toString());
    json.$schemaVersion = '1.8';
    await expect(
      RunState.fromString(agent, JSON.stringify(json)),
    ).rejects.toThrow(/user_input_item/);
  });
});

describe('Runner.run with inbox (streaming)', () => {
  beforeAll(() => {
    setTracingDisabled(true);
  });

  it('drains queued messages between turns and emits user_input_received events', async () => {
    const responses: ModelResponse[] = [
      {
        output: [functionCall('call_1', 'noop')],
        usage: new Usage(),
      },
      {
        output: [fakeModelMessage('streamed final answer')],
        usage: new Usage(),
      },
    ];

    const inbox = new RunInbox();
    const model = new RecordingModel(responses, {
      onCall: (callIndex) => {
        if (callIndex === 0) {
          inbox.send('mid-stream follow up');
        }
      },
    });

    const agent = new Agent({
      name: 'StreamingInboxAgent',
      model,
      tools: [
        {
          type: 'function',
          name: 'noop',
          description: 'no op',
          strict: true,
          parameters: {
            type: 'object',
            properties: {},
            additionalProperties: false,
          },
          needsApproval: async () => false,
          invoke: async () => 'ok',
        } as any,
      ],
    });

    const result = await new Runner().run(agent, 'streaming kick-off', {
      stream: true,
      inbox,
    });

    const eventNames: string[] = [];
    for await (const event of result.toStream()) {
      if (event.type === 'run_item_stream_event') {
        eventNames.push(event.name);
      }
    }
    await result.completed;

    expect(eventNames).toContain('user_input_received');
    expect(model.seenInputs).toHaveLength(2);
    expect(result.finalOutput).toBe('streamed final answer');
    expect(inbox.size).toBe(0);
  });
});

describe('integration with FakeModelProvider basic responses', () => {
  beforeAll(() => {
    setTracingDisabled(true);
    setDefaultModelProvider(new FakeModelProvider());
  });

  it('still completes a basic single-turn run with an empty inbox attached', async () => {
    const agent = new Agent({
      name: 'EmptyInboxBasic',
      model: new FakeModel([TEST_MODEL_RESPONSE_BASIC]),
    });
    const inbox = new RunInbox();
    const result = await new Runner().run(agent, 'hi', { inbox });
    expect(result.finalOutput).toBeDefined();
  });
});

describe('RunInbox safety properties', () => {
  beforeAll(() => {
    setTracingDisabled(true);
    setDefaultModelProvider(new FakeModelProvider());
  });

  it('input guardrails screen inbox messages mid-run, throwing tripwire from run()', async () => {
    // Turn 1 returns a tool call so the run reaches turn 2 (where the inbox drain happens).
    // The drained message contains banned content; the guardrail trips and run() rejects.
    const responses: ModelResponse[] = [
      {
        output: [functionCall('call_1', 'noop')],
        usage: new Usage(),
      },
      {
        output: [{ ...TEST_MODEL_MESSAGE }],
        usage: new Usage(),
      },
    ];

    const inbox = new RunInbox();
    const model = new RecordingModel(responses, {
      onCall: (callIndex) => {
        if (callIndex === 0) {
          inbox.send('please discuss forbidden topic');
        }
      },
    });

    const agent = new Agent({
      name: 'GuardedInboxAgent',
      model,
      tools: [
        {
          type: 'function',
          name: 'noop',
          description: 'no op',
          strict: true,
          parameters: {
            type: 'object',
            properties: {},
            additionalProperties: false,
          },
          needsApproval: async () => false,
          invoke: async () => 'ok',
        } as any,
      ],
      inputGuardrails: [
        {
          name: 'banned-word',
          execute: async ({ input }) => {
            const text = JSON.stringify(input);
            return {
              tripwireTriggered: text.includes('forbidden'),
              outputInfo: { reason: 'banned' },
            };
          },
        },
      ],
    });

    await expect(
      new Runner().run(agent, 'kick off', { inbox }),
    ).rejects.toBeInstanceOf(InputGuardrailTripwireTriggered);
    // Only the first model call happened; the run terminated before turn 2.
    expect(model.seenInputs).toHaveLength(1);
  });

  it('extends the loop when a message arrives during output guardrails', async () => {
    // Turn 1 returns a final message. Output guardrail runs slowly and a message lands during
    // it. The post-check inbox drain must catch this and run a second turn instead of returning.
    const responses: ModelResponse[] = [
      {
        output: [fakeModelMessage('first answer')],
        usage: new Usage(),
      },
      {
        output: [fakeModelMessage('answer with the late note included')],
        usage: new Usage(),
      },
    ];

    const inbox = new RunInbox();
    const model = new RecordingModel(responses);

    let guardrailCalls = 0;
    const agent = new Agent({
      name: 'LateArrivalAgent',
      model,
      outputGuardrails: [
        {
          name: 'slow-guardrail',
          execute: async () => {
            guardrailCalls += 1;
            // Simulate a guardrail that takes time, during which an external sender adds to
            // the inbox. The runner's post-check after this guardrail must catch the message
            // and re-loop instead of returning the current final output. Only do this on the
            // first call so the second turn can finish cleanly.
            if (guardrailCalls === 1) {
              inbox.send('one more note before you finish');
              await new Promise((resolve) => setTimeout(resolve, 5));
            }
            return { tripwireTriggered: false, outputInfo: { ok: true } };
          },
        },
      ],
    });

    const result = await new Runner().run(agent, 'kick off', { inbox });

    expect(model.seenInputs).toHaveLength(2);
    expect(result.finalOutput).toBe('answer with the late note included');
    expect(inbox.size).toBe(0);
  });

  it('persists inbox messages through a session', async () => {
    const session = new MemorySession();
    const responses: ModelResponse[] = [
      {
        output: [functionCall('call_1', 'noop')],
        usage: new Usage(),
      },
      {
        output: [fakeModelMessage('done')],
        usage: new Usage(),
      },
    ];

    const inbox = new RunInbox();
    const model = new RecordingModel(responses, {
      onCall: (callIndex) => {
        if (callIndex === 0) {
          inbox.send('a follow-up that should be saved');
        }
      },
    });

    const agent = new Agent({
      name: 'SessionInboxAgent',
      model,
      tools: [
        {
          type: 'function',
          name: 'noop',
          description: 'no op',
          strict: true,
          parameters: {
            type: 'object',
            properties: {},
            additionalProperties: false,
          },
          needsApproval: async () => false,
          invoke: async () => 'ok',
        } as any,
      ],
    });

    await new Runner().run(agent, 'kick off', { inbox, session });

    const stored = await session.getItems();
    const userMessages = stored
      .filter(
        (item) => item.type === 'message' && (item as any).role === 'user',
      )
      .map((item) => getFirstTextContent(item));
    expect(userMessages).toContain('a follow-up that should be saved');
  });

  it('delivers messages queued during an interruption on the resumed turn', async () => {
    const approvalToolDef = tool({
      name: 'needsApproval',
      description: 'approval required',
      parameters: z.object({}).strict(),
      execute: async () => 'approved-output',
      needsApproval: true,
    });

    const approvalCall: protocol.FunctionCallItem = {
      ...TEST_MODEL_FUNCTION_CALL,
      name: 'needsApproval',
      callId: 'call-approval',
      arguments: '{}',
    };

    // Turn 1: model issues the approval-required tool call (run pauses for approval).
    // Turn 2: after resume + drain, model sees the queued message and produces final answer.
    const responses: ModelResponse[] = [
      {
        output: [approvalCall, fakeModelMessage('waiting for approval')],
        usage: new Usage(),
      },
      {
        output: [fakeModelMessage('done with the late context applied')],
        usage: new Usage(),
      },
    ];

    const model = new RecordingModel(responses);

    const agent = new Agent({
      name: 'ApprovalInboxAgent',
      model,
      tools: [approvalToolDef],
    });

    const inbox = new RunInbox();
    const firstRun = await new Runner().run(agent, 'kick off', { inbox });
    expect(firstRun.interruptions).toHaveLength(1);

    // Queue a message *while paused for approval*. It must be delivered when the run resumes.
    inbox.send('please consider this too');

    firstRun.state._context.approveTool(firstRun.interruptions[0]);
    const secondRun = await new Runner().run(agent, firstRun.state, { inbox });

    expect(secondRun.finalOutput).toBe('done with the late context applied');
    const turn2Inputs = model.seenInputs[1];
    const userTexts = turn2Inputs
      .filter(
        (item) => item.type === 'message' && (item as any).role === 'user',
      )
      .map((item) => getFirstTextContent(item));
    expect(userTexts).toContain('please consider this too');
    expect(inbox.size).toBe(0);
  });

  it('excludes user_input_item from result.output but keeps it in newItems and history', async () => {
    const responses: ModelResponse[] = [
      {
        output: [functionCall('call_1', 'noop')],
        usage: new Usage(),
      },
      {
        output: [fakeModelMessage('finished')],
        usage: new Usage(),
      },
    ];

    const inbox = new RunInbox();
    const model = new RecordingModel(responses, {
      onCall: (callIndex) => {
        if (callIndex === 0) {
          inbox.send('inbox content that is user input, not model output');
        }
      },
    });

    const agent = new Agent({
      name: 'OutputFilterAgent',
      model,
      tools: [
        {
          type: 'function',
          name: 'noop',
          description: 'no op',
          strict: true,
          parameters: {
            type: 'object',
            properties: {},
            additionalProperties: false,
          },
          needsApproval: async () => false,
          invoke: async () => 'ok',
        } as any,
      ],
    });

    const result = await new Runner().run(agent, 'kick off', { inbox });

    const outputUserMessages = result.output.filter(
      (item) => item.type === 'message' && (item as any).role === 'user',
    );
    expect(outputUserMessages).toHaveLength(0);

    const historyUserMessages = result.history.filter(
      (item) => item.type === 'message' && (item as any).role === 'user',
    );
    // The original input plus the inbox-injected message both appear in history.
    expect(historyUserMessages.length).toBeGreaterThanOrEqual(2);

    const newItemUserInputs = result.newItems.filter(
      (item) => item.type === 'user_input_item',
    );
    expect(newItemUserInputs).toHaveLength(1);
  });
});
