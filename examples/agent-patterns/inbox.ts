/**
 * RunInbox demo: queues a mid-run user message into a `RunInbox` and watches the agent pick it
 * up at the next turn boundary without aborting the run. The `plan_day` tool is intentionally
 * slow so a human watching the demo has time to type while a tool call is in flight.
 *
 * Suggested demo: type "I just realized I'm bringing my dog! Make activities dog-friendly."
 * after day 2 finishes. Days 3–5 should reflect the new constraint without any restart.
 */
import readline from 'node:readline';
import { Agent, RunInbox, run, tool } from '@openai/agents';
import chalk from 'chalk';
import { z } from 'zod';

// Slow on purpose so a human watching the demo has time to type during a tool call.
const TOOL_LATENCY_MS = 6000;

const planDay = tool({
  name: 'plan_day',
  description:
    'Plan activities for one day of a trip. Call this exactly once per day, in order, before producing the final itinerary.',
  parameters: z.object({
    day: z.number().int().min(1).describe('Day number, starting from 1.'),
    theme: z
      .string()
      .describe(
        'Short summary of the theme/constraints for this day (e.g. "food tour", "dog-friendly outdoor", "rainy-day backup").',
      ),
  }),
  execute: async ({ day, theme }) => {
    await new Promise((resolve) => setTimeout(resolve, TOOL_LATENCY_MS));
    return `Day ${day} (${theme}): three suggested stops with timings.`;
  },
});

const agent = new Agent({
  name: 'TripPlanner',
  instructions: `
You are planning a multi-day trip. For each day from 1 through the requested
total, call the \`plan_day\` tool exactly once, IN ORDER, before producing any
prose. Pass a short \`theme\` string that captures the user's current
constraints. After all days are planned, write a brief itinerary acknowledging
those constraints.

If the user gives you new constraints partway through (you'll see them as
fresh user messages between tool calls), incorporate them into the remaining
days and call them out explicitly in your final itinerary. Do not redo days
that were already planned.
  `.trim(),
  tools: [planDay],
  model: 'gpt-4.1',
});

const inbox = new RunInbox();
const rl = readline.createInterface({ input: process.stdin });

rl.on('line', (line) => {
  const text = line.trim();
  if (!text) return;
  inbox.send(text);
  console.log(
    chalk.yellow(
      `\n>> queued for agent: "${text}" (inbox size: ${inbox.size})\n`,
    ),
  );
});

async function main() {
  console.log(chalk.bgCyan('  ● RunInbox demo: trip planner  \n'));
  console.log(
    chalk.dim(
      'Type any message and press Enter while the agent is working.\n' +
        'It will be queued and delivered at the next turn boundary.\n',
    ),
  );

  const prompt =
    'Plan a 5-day Tokyo trip. Plan days 1 through 5 individually using plan_day, in order, then summarize.';

  console.log(`${chalk.bold('Prompt:')} ${prompt}\n`);

  const result = await run(agent, prompt, { stream: true, inbox });

  for await (const event of result.toStream()) {
    if (event.type !== 'run_item_stream_event') continue;

    switch (event.name) {
      case 'tool_called': {
        const item = event.item as any;
        const args = item.rawItem?.arguments ?? '{}';
        console.log(chalk.cyan(`[tool] ${item.toolName}(${args})`));
        break;
      }
      case 'tool_output': {
        const item = event.item as any;
        console.log(chalk.gray(`       → ${item.output}`));
        break;
      }
      case 'user_input_received': {
        // Fires the moment the runner drained the inbox into the turn input.
        const item = event.item as any;
        console.log(chalk.magenta(`[inbox delivered] "${item.content}"`));
        break;
      }
      case 'message_output_created': {
        const item = event.item as any;
        console.log(chalk.green(`\n[assistant]\n${item.content}\n`));
        break;
      }
      default:
        break;
    }
  }

  await result.completed;
  console.log(chalk.bgCyan('  ● run complete  '));
  rl.close();
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
