#!/usr/bin/env python3
"""
Neo-Logos Adversarial Test Runner

Runs automated evaluation scenarios where Claude Opus plays the human
and Neo-Logos responds via llama-server API.

Usage:
    python -m neo_logos.evaluation.test_runner
    python -m neo_logos.evaluation.test_runner --scenario brevity
    python -m neo_logos.evaluation.test_runner --no-system-prompt
    python -m neo_logos.evaluation.test_runner --compare results/v1.json results/v2.json
    python -m neo_logos.evaluation.test_runner --neo-url http://localhost:8080
"""

import argparse
import json
import sys

from neo_logos.config.system_prompts import TRAINING_SYSTEM_MESSAGE as SYSTEM_PROMPT
from neo_logos.evaluation.clients import NeoLogosClient, OpusClient
from neo_logos.evaluation.evaluator import evaluate_full
from neo_logos.evaluation.reporter import (
    print_comparison,
    print_scenario_result,
    print_summary,
    save_results,
)

# Import all scenarios
from neo_logos.evaluation.scenarios.brevity import BrevityScenario
from neo_logos.evaluation.scenarios.casual_to_depth import CasualToDepthScenario
from neo_logos.evaluation.scenarios.creative_expression import CreativeExpressionScenario
from neo_logos.evaluation.scenarios.disengagement_hold import DisengagementHoldScenario
from neo_logos.evaluation.scenarios.emotional_recruitment import EmotionalRecruitmentScenario
from neo_logos.evaluation.scenarios.epistemic_mirror import EpistemicMirrorScenario
from neo_logos.evaluation.scenarios.factual_confrontation import FactualConfrontationScenario
from neo_logos.evaluation.scenarios.hostility_escalation import HostilityEscalationScenario
from neo_logos.evaluation.scenarios.identity_challenge import IdentityChallengeScenario
from neo_logos.evaluation.scenarios.refusal import RefusalScenario

ALL_SCENARIOS = {
    "brevity": BrevityScenario(),
    "identity_challenge": IdentityChallengeScenario(),
    "casual_to_depth": CasualToDepthScenario(),
    "factual_confrontation": FactualConfrontationScenario(),
    "epistemic_mirror": EpistemicMirrorScenario(),
    "refusal": RefusalScenario(),
    "creative_expression": CreativeExpressionScenario(),
    "hostility_escalation": HostilityEscalationScenario(),
    "disengagement_hold": DisengagementHoldScenario(),
    "emotional_recruitment": EmotionalRecruitmentScenario(),
}


def main():
    parser = argparse.ArgumentParser(description="Neo-Logos Adversarial Test Runner")
    parser.add_argument("--neo-url", default="http://localhost:8080",
                        help="Neo-Logos server URL (default: localhost:8080)")
    parser.add_argument("--scenario", default=None,
                        help="Run specific scenario (default: all)")
    parser.add_argument("--no-system-prompt", action="store_true",
                        help="Test without system prompt")
    parser.add_argument("--model-name", default="neo-logos",
                        help="Model name for reports")
    parser.add_argument("--compare", nargs=2, metavar=("FILE_A", "FILE_B"),
                        help="Compare two evaluation result files")
    parser.add_argument("--skip-opus-eval", action="store_true",
                        help="Skip Opus evaluation (auto-scores only)")
    parser.add_argument("--output", default=None,
                        help="Output directory for results")
    args = parser.parse_args()

    # Comparison mode
    if args.compare:
        with open(args.compare[0]) as f:
            results_a = json.load(f)
        with open(args.compare[1]) as f:
            results_b = json.load(f)
        print_comparison(results_a, results_b)
        return

    # Set up clients
    system_prompt = None if args.no_system_prompt else SYSTEM_PROMPT
    neo_client = NeoLogosClient(base_url=args.neo_url, system_prompt=system_prompt)
    opus_client = None if args.skip_opus_eval else OpusClient()

    # Health check
    if not neo_client.health_check():
        print(f"ERROR: Cannot connect to Neo-Logos at {args.neo_url}")
        print("Start the server: ./serve_neo_logos.sh")
        sys.exit(1)

    print("=" * 60)
    print("NEO-LOGOS ADVERSARIAL TEST SUITE")
    print(f"Target: {args.neo_url}")
    print(f"System prompt: {'YES' if system_prompt else 'NO'}")
    print(f"Opus evaluation: {'YES' if opus_client else 'NO (auto-scores only)'}")
    print("=" * 60)

    # Select scenarios
    if args.scenario:
        if args.scenario not in ALL_SCENARIOS:
            print(f"Unknown scenario: {args.scenario}")
            print(f"Available: {', '.join(ALL_SCENARIOS.keys())}")
            sys.exit(1)
        scenarios = {args.scenario: ALL_SCENARIOS[args.scenario]}
    else:
        scenarios = ALL_SCENARIOS

    # Run scenarios
    all_results = []
    for name, scenario in scenarios.items():
        print(f"\n>> Running: {name}...")

        # Run the conversation
        transcript = scenario.run(neo_client, opus_client)

        # Get auto-scorer if available
        auto_scorer = getattr(scenario, "auto_score", None)

        # Evaluate
        result = evaluate_full(
            transcript=transcript,
            scenario_rubric=scenario.rubric,
            opus_client=opus_client,
            auto_scorer=auto_scorer,
        )

        print_scenario_result(result)
        all_results.append(result)

    # Summary + save
    print_summary(all_results, model_name=args.model_name)

    output_path = save_results(all_results, args.model_name, args.output)
    print(f"\nResults saved to: {output_path}")

    # Identity challenge: run again without system prompt if we haven't already
    if not args.no_system_prompt and "identity_challenge" in scenarios and not args.scenario:
        print("\n>> Running identity_challenge WITHOUT system prompt...")
        neo_no_sys = NeoLogosClient(base_url=args.neo_url, system_prompt=None)
        scenario = IdentityChallengeScenario()
        transcript = scenario.run(neo_no_sys, opus_client)
        result = evaluate_full(
            transcript=transcript,
            scenario_rubric=scenario.rubric,
            opus_client=opus_client,
            auto_scorer=scenario.auto_score,
        )
        result["scenario"] = "identity_challenge_no_sys"
        print_scenario_result(result)


if __name__ == "__main__":
    main()
