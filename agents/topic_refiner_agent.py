from agents.topic_refiner import refine_all_topics


def run_topic_refiner(domain: str, topics: str, level: str = "M.Tech") -> dict:
    """
    Refines generic topics into publication-style research directions.
    """

    print(f"Topic Refiner running for domain: {domain}")

    refined = refine_all_topics(topics, domain, level)

    print(f"Topic Refiner done.")
    return {
        "domain": domain,
        "level": level,
        "refined_topics": refined
    }