import sys
import os
from crew import AINewsCrew

if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    crew = AINewsCrew()
    result = crew.kickoff()
    print("\n===== AI News Article of the Week =====\n")
    print(result)
