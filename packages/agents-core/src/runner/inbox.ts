import { Agent } from '../agent';
import type { InputGuardrailDefinition } from '../guardrail';
import type { RunInbox } from '../inbox';
import { RunUserInputItem } from '../items';
import type { RunState } from '../runState';
import {
  buildInputGuardrailDefinitions,
  runInputGuardrails,
} from './guardrails';

/**
 * Moves any messages waiting in the inbox into the run's generated-items list so the next call
 * to `prepareTurn` includes them in the model input.
 *
 * Called at two boundaries:
 *  1. The top of every `next_step_run_again` iteration, so messages queued mid-run reach the
 *     model on the next turn.
 *  2. Just before the loop would honor `next_step_final_output`, where a non-empty inbox
 *     converts the step into another `next_step_run_again` so messages received during the final
 *     model call are not dropped.
 *
 * The chosen drain location matters: pushing onto `_generatedItems` (rather than `_originalInput`)
 * preserves chronological order in `result.history`, and the items are picked up automatically by
 * `prepareModelInputItems`, the `ServerConversationTracker`, and session persistence.
 *
 * Input guardrails configured on the runner or active agent are executed against the drained
 * messages before they are appended to state. This keeps the screening boundary symmetric with
 * how initial user input is handled — content arriving via the inbox cannot bypass policy
 * checks. A tripwire throws `InputGuardrailTripwireTriggered`; a guardrail crash throws
 * `GuardrailExecutionError`. In either case the offending batch is dropped (not partially
 * persisted) so resumed runs do not contain unscreened content. For simplicity v1 runs all
 * configured input guardrails as blocking against the drained batch; honoring `runInParallel`
 * for inbox messages can be added later without changing the public surface.
 *
 * Returns the items appended (or `[]`) so streaming callers can emit corresponding events.
 */
export async function drainInboxIntoState(
  state: RunState<any, Agent<any, any>>,
  inbox: RunInbox | undefined,
  runnerInputGuardrailDefs: InputGuardrailDefinition[],
): Promise<RunUserInputItem[]> {
  if (!inbox || inbox.size === 0) {
    return [];
  }
  const messages = inbox.drain();
  if (messages.length === 0) {
    return [];
  }
  const agent = state._currentAgent as Agent<any, any>;

  const guardrailDefs = buildInputGuardrailDefinitions(
    state,
    runnerInputGuardrailDefs,
  );
  if (guardrailDefs.length > 0) {
    // Throws on tripwire / execution failure. Messages are dropped (already removed from the
    // inbox via drain, but never appended to state) so a caller catching the exception and
    // resuming from `state` does not see unscreened content in history.
    await runInputGuardrails(state, guardrailDefs, {
      input: messages,
      rollbackTurnOnError: false,
    });
  }

  const items: RunUserInputItem[] = [];
  for (const message of messages) {
    const item = new RunUserInputItem(message, agent);
    state._generatedItems.push(item);
    items.push(item);
  }
  return items;
}
