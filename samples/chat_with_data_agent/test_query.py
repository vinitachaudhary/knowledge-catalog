"""Send a test query to a deployed Chat with Data agent."""

import argparse
import re
import sys

import vertexai
from vertexai._genai import types


def parse_resource_id(resource_id: str) -> tuple[str, str]:
    """Return (project, location) from a full reasoning engine resource name."""
    m = re.match(
        r"projects/([^/]+)/locations/([^/]+)/reasoningEngines/[^/]+",
        resource_id,
    )
    if not m:
        raise ValueError(
            "Expected format: projects/PROJECT/locations/LOCATION/reasoningEngines/ID"
            f", got: {resource_id}"
        )
    return m.group(1), m.group(2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Send a query to a deployed Chat with Data agent."
    )
    parser.add_argument(
        "resource_id",
        help="Full resource name: projects/PROJECT/locations/LOCATION/reasoningEngines/ID",
    )
    parser.add_argument("query", help="Query to send to the agent")
    args = parser.parse_args()

    project, location = parse_resource_id(args.resource_id)
    print(f"Project : {project}")
    print(f"Location: {location}")
    print(f"Resource: {args.resource_id}")
    print(f"Query   : {args.query}\n")

    client = vertexai.Client(project=project, location=location)

    config = types.QueryAgentEngineConfig(
        class_method="stream_query",
        input={"message": args.query, "user_id": "vinitac"},
    )

    import json

    for event in client.agent_engines._stream_query(
        name=args.resource_id, config=config
    ):
        try:
            data = json.loads(event.body)
        except (json.JSONDecodeError, AttributeError):
            continue
        if "error_message" in data:
            print(f"[error] {data['error_message']}", flush=True)
            continue
        for part in data.get("content", {}).get("parts", []):
            text = part.get("text")
            if text:
                print(text, end="", flush=True)

    print()


if __name__ == "__main__":
    main()
