"""Run the actual loop's transition statements with small fake agents."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

SOURCE = Path(__file__).resolve().parents[1] / 'ai.py'

class TerminalTransitionTests(unittest.TestCase):
    def test_terminal_record_precedes_reset(self):
        module = ast.parse(SOURCE.read_text())
        loop = next(n for n in module.body if isinstance(n, ast.While))
        start = next(i for i, n in enumerate(loop.body) if isinstance(n, ast.Assign) and
                     isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'done')
        stop = next(i for i, n in enumerate(loop.body) if isinstance(n, ast.Expr) and
                    isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute) and
                    isinstance(n.value.func.value, ast.Name) and n.value.func.value.id == 'screen')
        records = []
        class Agent:
            health = 0
            x = y = 0
            level = 1
            replay_buffer = SimpleNamespace(add=records.append)
            def get_state(self, zombies): return self.health
            def train(self): pass
            def reset(self): self.health = 100
        agent = Agent()
        ns = dict(ai_savior=agent, zombies=[], st='before', act=3,
                  CENTER_X=0, CENTER_Y=0, CENTER_RADIUS=200,
                  math=__import__('math'), start_level=lambda level: ['new-episode'])
        exec(compile(ast.Module(body=loop.body[start:stop], type_ignores=[]), str(SOURCE), 'exec'), ns)
        self.assertEqual(len(records), 1)
        state, action, reward, next_state, done = records[0]
        self.assertEqual(next_state, 0)
        self.assertTrue(done)
        self.assertLess(reward, -90)
        self.assertEqual(agent.health, 100)

if __name__ == '__main__':
    unittest.main()
