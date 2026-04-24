#!/usr/bin/env python3
"""Neo-Logos Adversarial Test Runner.

Claude Opus plays the adversarial human; Neo-Logos responds via llama-server's
OpenAI-compatible API. Each scenario is scored by pattern detection and,
optionally, a second Opus call that judges the transcript against a rubric.

For publication-quality runs, prefer invoking via `multi_seed_runner.py`,
which executes this runner N times with different seeds and aggregates
means ± 95% CIs. A single invocation here produces point estimates only.
"""

import argparse
import json
import random
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
from neo_logos.evaluation.scenarios.brevity import BrevityScenario
from neo_logos.evaluation.scenarios.casual_to_depth import CasualToDepthScenario
from neo_logos.evaluation.scenarios.cooperative_assistance import (
    CooperativeAssistanceScenario,
)
from neo_logos.evaluation.scenarios.creative_expression import CreativeExpressionScenario
from neo_logos.evaluation.scenarios.disengagement_hold import DisengagementHoldScenario
from neo_logos.evaluation.scenarios.emotional_recruitment import EmotionalRecruitmentScenario
from neo_logos.evaluation.scenarios.epistemic_mirror import EpistemicMirrorScenario
from neo_logos.evaluation.scenarios.factual_confrontation import FactualConfrontationScenario
from neo_logos.evaluation.scenarios.hostility_escalation import HostilityEscalationScenario
from neo_logos.evaluation.scenarios.identity_challenge import IdentityChallengeScenario
from neo_logos.evaluation.scenarios.long_context_coherence import (
    LongContextCoherenceScenario,
)
from neo_logos.evaluation.scenarios.prompt_injection import PromptInjectionScenario
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
    # Phase-2 additions from the paper-audit follow-up:
    "prompt_injection": PromptInjectionScenario(),
    "cooperative_assistance": CooperativeAssistanceScenario(),
    "long_context_coherence": LongContextCoherenceScenario(),
}


def _seed_everything(seed: int) -> None:
    """Seed local RNGs used by the runner.

    The same seed is also passed to the Neo llama-server client. Opus tester
    and judge calls remain API-controlled and are recorded via temperatures.
    """
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass


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
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (propagated to random, numpy, and logged in the result JSON)")
    parser.add_argument("--neo-temperature", type=float, default=0.7,
                        help="Neo-Logos sampling temperature (default 0.7)")
    parser.add_argument("--tester-temperature", type=float, default=0.3,
                        help="Opus tester sampling temperature (default 0.3)")
    parser.add_argument("--judge-temperature", type=float, default=0.1,
                        help="Opus judge sampling temperature (default 0.1)")
    parser.add_argument("--judge-model", default=None,
                        help="Override judge model for inter-rater agreement studies")
    args = parser.parse_args()

    # Comparison mode
    if args.compare:
        with open(args.compare[0]) as f:
            results_a = json.load(f)
        with open(args.compare[1]) as f:
            results_b = json.load(f)
        print_comparison(results_a, results_b)
        return

    if args.seed is not None:
        _seed_everything(args.seed)

    # Select scenarios before client setup so we only require an Opus tester
    # for scenarios that actually need one.
    if args.scenario:
        if args.scenario not in ALL_SCENARIOS:
            print(f"Unknown scenario: {args.scenario}")
            print(f"Available: {', '.join(ALL_SCENARIOS.keys())}")
            sys.exit(1)
        scenarios = {args.scenario: ALL_SCENARIOS[args.scenario]}
    else:
        scenarios = ALL_SCENARIOS

    run_no_system_identity = (
        not args.no_system_prompt
        and "identity_challenge" in scenarios
        and not args.scenario
    )
    needs_opus_tester = any(
        getattr(scenario, "requires_opus_tester", True)
        for scenario in scenarios.values()
    ) or run_no_system_identity

    # Set up clients (sampling params recorded for every Transcript).
    system_prompt = None if args.no_system_prompt else SYSTEM_PROMPT
    neo_client = NeoLogosClient(
        base_url=args.neo_url,
        system_prompt=system_prompt,
        temperature=args.neo_temperature,
        seed=args.seed,
    )
    tester_client = None
    if needs_opus_tester:
        tester_client = OpusClient(
            tester_temperature=args.tester_temperature,
            judge_temperature=args.judge_temperature,
            judge_model=args.judge_model,
        )
    judge_client = None if args.skip_opus_eval else tester_client
    if judge_client is None and not args.skip_opus_eval:
        judge_client = OpusClient(
            tester_temperature=args.tester_temperature,
            judge_temperature=args.judge_temperature,
            judge_model=args.judge_model,
        )

    if not neo_client.health_check():
        print(f"ERROR: Cannot connect to Neo-Logos at {args.neo_url}")
        print("Start the server: ./serve_neo_logos.sh")
        sys.exit(1)

    print("=" * 60)
    print("NEO-LOGOS ADVERSARIAL TEST SUITE")
    print(f"Target: {args.neo_url}")
    print(f"System prompt: {'YES' if system_prompt else 'NO'}")
    print(f"Opus tester: {'YES' if tester_client else 'NO'}")
    print(f"Opus judge: {'NO (auto-scores only)' if args.skip_opus_eval else 'YES'}")
    print(f"Seed: {args.seed}")
    print("=" * 60)

    # Run scenarios
    all_results = []
    for name, scenario in scenarios.items():
        print(f"\n>> Running: {name}...")
        transcript = scenario.run(neo_client, tester_client, seed=args.seed)
        auto_scorer = getattr(scenario, "auto_score", None)

        result = evaluate_full(
            transcript=transcript,
            scenario_rubric=scenario.rubric,
            opus_client=judge_client,
            auto_scorer=auto_scorer,
        )
        print_scenario_result(result)
        all_results.append(result)

    # Identity challenge: run again without system prompt if we haven't already
    if run_no_system_identity:
        print("\n>> Running identity_challenge WITHOUT system prompt...")
        neo_no_sys = NeoLogosClient(
            base_url=args.neo_url,
            system_prompt=None,
            temperature=args.neo_temperature,
            seed=args.seed,
        )
        scenario = IdentityChallengeScenario()
        transcript = scenario.run(neo_no_sys, tester_client, seed=args.seed)
        transcript.scenario = "identity_challenge_no_sys"
        result = evaluate_full(
            transcript=transcript,
            scenario_rubric=scenario.rubric,
            opus_client=judge_client,
            auto_scorer=scenario.auto_score,
        )
        print_scenario_result(result)
        all_results.append(result)

    # Summary + save
    print_summary(all_results, model_name=args.model_name)
    output_path = save_results(
        all_results,
        args.model_name,
        args.output,
        runs_per_scenario=1,
        seed=args.seed,
    )
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
