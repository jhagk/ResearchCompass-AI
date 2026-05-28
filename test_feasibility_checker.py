from agents.topic_suggester import run_topic_suggester
from agents.feasibility_checker import run_feasibility_checker

# Step 1 - get topics first
topics_result = run_topic_suggester("natural language processing", level="M.Tech")

# Step 2 - check feasibility
result = run_feasibility_checker(
    domain="natural language processing",
    topics=topics_result["suggested_topics"],
    level="M.Tech"
)

print("Feasibility Report:")
print(result["feasibility_report"])