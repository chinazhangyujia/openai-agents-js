import { Agent, RunInbox, run } from '@openai/agents';

const agent = new Agent({
  name: 'Planner',
  instructions: "Plan the user's deep work session. Be thorough.",
});

// The caller holds the inbox; the runner observes it. Mirrors the AbortSignal pattern.
const inbox = new RunInbox();

const promise = run(agent, 'plan a 30-minute deep work session', { inbox });

// Some time later, from another async context (a websocket handler, an HTTP endpoint, etc.):
inbox.send('also remind me to drink water halfway through');

const result = await promise;

console.log(result.finalOutput);
