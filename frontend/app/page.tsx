"use client";

import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type KeyboardEvent,
} from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

// Single source of truth for the backend base URL. Change this one line
// instead of hunting down every fetch() call when the dev server port changes.
const API_BASE_URL = "http://127.0.0.1:8006";

/* ─────────────────────────── types ─────────────────────────── */

type BurstinessLevel = "low" | "medium" | "high";

type InfrastructureAssumptions = {
  gpuUtilizationPct: number;
  enterpriseApiDiscountPct: number;
  burstiness: BurstinessLevel;
  reservedFailoverCapacityPct: number;
};

const defaultAssumptions: InfrastructureAssumptions = {
  gpuUtilizationPct: 70,
  enterpriseApiDiscountPct: 0,
  burstiness: "medium",
  reservedFailoverCapacityPct: 15,
};

type DeploymentPresetId =
  | "startup_mvp"
  | "enterprise_saas"
  | "internal_copilot"
  | "coding_agent";

const DEPLOYMENT_PRESETS: Record<
  DeploymentPresetId,
  { label: string; values: InfrastructureAssumptions }
> = {
  startup_mvp: {
    label: "Startup MVP",
    values: { gpuUtilizationPct: 55, enterpriseApiDiscountPct: 0, burstiness: "medium", reservedFailoverCapacityPct: 10 },
  },
  enterprise_saas: {
    label: "Enterprise SaaS",
    values: { gpuUtilizationPct: 72, enterpriseApiDiscountPct: 20, burstiness: "medium", reservedFailoverCapacityPct: 20 },
  },
  internal_copilot: {
    label: "Internal AI Copilot",
    values: { gpuUtilizationPct: 68, enterpriseApiDiscountPct: 10, burstiness: "low", reservedFailoverCapacityPct: 15 },
  },
  coding_agent: {
    label: "Coding Agent Platform",
    values: { gpuUtilizationPct: 60, enterpriseApiDiscountPct: 5, burstiness: "high", reservedFailoverCapacityPct: 25 },
  },
};

const BURSTINESS_MULTIPLIERS: Record<BurstinessLevel, number> = {
  low: 1.0,
  medium: 1.15,
  high: 1.35,
};

function getAssumptionImpactSummary(assumptions: InfrastructureAssumptions) {
  const burstinessMultiplier = BURSTINESS_MULTIPLIERS[assumptions.burstiness];
  const utilizationFraction = assumptions.gpuUtilizationPct / 100;
  const failoverMultiplier = 1 + assumptions.reservedFailoverCapacityPct / 100;
  const effectiveCapacityMultiplier = (burstinessMultiplier * failoverMultiplier) / utilizationFraction;
  return {
    burstinessMultiplier,
    failoverMultiplier,
    utilizationFraction,
    effectiveCapacityMultiplier,
    formulaDisplay: `${burstinessMultiplier} × ${failoverMultiplier.toFixed(2)} / ${utilizationFraction.toFixed(2)}`,
  };
}

type ComplexityLevel = "low" | "medium" | "high";
type ContextComplexity = "short-context" | "medium-context" | "long-context";
type ModelTier = "budget" | "premium" | "frontier";

type ViableModelTiers = { budget: boolean; premium: boolean; frontier: boolean };

type CapabilityAwareRecommendation = {
  recommendedOption: string;
  rationale: string;
  migrationTrigger: string;
  capabilityConstraints: string[];
  viableModelTiers: ViableModelTiers;
  selectedApiTier: ModelTier | null;
  preferredModelFamily: string | null;
};

type FieldProvenance = {
  source: "user_provided" | "assumed" | "missing";
  display_value: string | null;
  internal_value: string | number | null;
};

type PlanStructuredAssumptions = {
  workload?: Record<string, FieldProvenance>;
  complexity?: Record<string, FieldProvenance>;
  operational?: {
    gpu_utilization_pct: number;
    enterprise_api_discount_pct: number;
    burstiness_factor: string;
    failover_reserve_pct: number;
  };
};

type PlanResponse = {
  assistant_message: string;
  workload_summary?: string;
  structured_assumptions: PlanStructuredAssumptions;
  missing_fields: string[];
  clarification_questions: string[];
  assumed_fields: string[];
  ready_to_simulate: boolean;
};

type PlannerTurn = { id: string; role: "user" | "planner"; content: string };

function getFieldInternal(cell: FieldProvenance | number | string | null | undefined): number | string | null {
  if (cell == null) return null;
  if (typeof cell === "object" && "internal_value" in cell) return cell.internal_value;
  return cell as number | string;
}

function getPlannerStatus(plan: PlanResponse | null): { label: string; tone: "collecting" | "ready" | "estimated" } {
  if (!plan) return { label: "Collecting assumptions", tone: "collecting" };
  if (plan.ready_to_simulate && plan.assumed_fields.length > 0) return { label: "Assumptions include estimates", tone: "estimated" };
  if (plan.ready_to_simulate) return { label: "Ready to simulate", tone: "ready" };
  return { label: "Collecting assumptions", tone: "collecting" };
}

// Maps the real ADK verdict string to a status pill. Driven by adkLastVerdict,
// which is set the instant a response arrives — this is what makes the pill
// change immediately on arrival, distinguishing "judge needs more from you"
// from "this is your final answer" the moment the response lands, rather than
// both outcomes looking identical while loading.
function getVerdictStatus(verdict: string | null, loading: boolean): { label: string; tone: "collecting" | "ready" | "estimated" | "needs_user" } {
  if (loading) return { label: "Thinking...", tone: "collecting" };
  if (verdict === "pass") return { label: "Simulation complete", tone: "ready" };
  if (verdict === "needs_user" || verdict === "retry") return { label: "Needs clarification", tone: "needs_user" };
  if (verdict === "infeasible") return { label: "Not possible — see explanation", tone: "needs_user" };
  return { label: "Collecting assumptions", tone: "collecting" };
}

// Adapts the flat ADK workload_spec (plus the Parsing Agent's field_confidence
// map) into the FieldProvenance-wrapped shape AssumptionsTable expects. Fields
// listed in field_confidence were estimated with no real basis in the user's
// text; everything else present is treated as user-provided/computed.
function adkSpecToStructuredAssumptions(
  spec: Record<string, any> | null | undefined,
): PlanStructuredAssumptions {
  if (!spec) return { workload: {}, complexity: {} };
  const fieldConfidence: Record<string, string> = spec.field_confidence ?? {};

  function cell(fieldKey: string, displayFn?: (v: any) => string): FieldProvenance {
    const raw = spec?.[fieldKey];
    if (raw === undefined || raw === null || raw === "" || raw === 0) {
      return { source: "missing", display_value: null, internal_value: null };
    }
    const isEstimated = fieldKey in fieldConfidence;
    return {
      source: isEstimated ? "assumed" : "user_provided",
      display_value: displayFn ? displayFn(raw) : String(raw),
      internal_value: raw,
    };
  }

  return {
    workload: {
      monthly_queries: cell("monthly_queries", (v) => Number(v).toLocaleString()),
      input_tokens_per_query: cell("input_tokens_per_query", (v) => `${v} tokens`),
      output_tokens_per_query: cell("output_tokens_per_query", (v) => `${v} tokens`),
      // ADK emits latency_sla as a string ("real-time"/"interactive"/"batch"),
      // not a millisecond value — display the string, no internal_value to compute with.
      latency_sla_ms: spec.latency_sla
        ? { source: "user_provided", display_value: String(spec.latency_sla), internal_value: String(spec.latency_sla) }
        : { source: "missing", display_value: null, internal_value: null },
    },
    complexity: {
      reasoning_complexity: cell("reasoning_complexity"),
      context_complexity: cell("context_complexity"),
      hallucination_sensitivity: cell("hallucination_sensitivity"),
    },
  };
}

const PLANNER_FIELD_LABELS: Record<string, string> = {
  monthly_queries: "Monthly query volume",
  input_tokens_per_query: "Input context estimate",
  output_tokens_per_query: "Output length estimate",
  latency_sla_ms: "Response time target",
  reasoning_complexity: "Reasoning complexity",
  context_complexity: "Context size profile",
  hallucination_sensitivity: "Accuracy sensitivity",
};

function AssumptionsTable({ assumptions }: { assumptions: PlanStructuredAssumptions }) {
  const workloadFields = ["monthly_queries", "input_tokens_per_query", "output_tokens_per_query", "latency_sla_ms"] as const;
  const complexityFields = ["reasoning_complexity", "context_complexity", "hallucination_sensitivity"] as const;

  function renderRow(fieldKey: string, cell: FieldProvenance | undefined) {
    const label = PLANNER_FIELD_LABELS[fieldKey] ?? fieldKey;
    const display = cell?.display_value ?? "—";
    const source = cell?.source ?? "missing";
    return (
      <tr key={fieldKey} style={{ borderTop: "1px solid rgba(255,255,255,0.05)" }}>
        <td style={{ padding: "8px 0", fontSize: T.fontSmall, color: T.textMuted, fontFamily: "'DM Mono', monospace" }}>{label}</td>
        <td style={{ padding: "8px 0", fontSize: T.fontBody, color: "#E5E7EB", fontFamily: "'DM Mono', monospace" }}>{display}</td>
        <td style={{ padding: "8px 0", paddingLeft: 8, textAlign: "right" }}>
          {source === "user_provided" && (
            <span style={{ fontSize: T.fontMicro, letterSpacing: "0.12em", color: "#7C3AED", fontFamily: "'DM Mono', monospace", background: "rgba(124,58,237,0.12)", padding: "2px 6px", borderRadius: 3, border: "1px solid rgba(124,58,237,0.25)" }}>PROVIDED</span>
          )}
          {source === "assumed" && (
            <span style={{ fontSize: T.fontMicro, letterSpacing: "0.12em", color: "#F59E0B", fontFamily: "'DM Mono', monospace", background: "rgba(245,158,11,0.1)", padding: "2px 6px", borderRadius: 3, border: "1px solid rgba(245,158,11,0.25)" }}>ESTIMATED</span>
          )}
          {source === "missing" && (
            <span style={{ fontSize: T.fontMicro, letterSpacing: "0.12em", color: T.textDim, fontFamily: "'DM Mono', monospace" }}>MISSING</span>
          )}
        </td>
      </tr>
    );
  }

  return (
    <div style={{ marginTop: 16 }}>
      <div style={{ marginBottom: 16 }}>
        <div style={{ fontSize: T.fontMicro, letterSpacing: "0.18em", color: T.textMuted, fontFamily: "'DM Mono', monospace", marginBottom: 8 }}>WORKLOAD</div>
        <table style={{ width: "100%", borderCollapse: "collapse" }}>
          <tbody>{workloadFields.map((f) => renderRow(f, assumptions.workload?.[f]))}</tbody>
        </table>
      </div>
      <div>
        <div style={{ fontSize: T.fontMicro, letterSpacing: "0.18em", color: T.textMuted, fontFamily: "'DM Mono', monospace", marginBottom: 8 }}>COMPLEXITY PROFILE</div>
        <table style={{ width: "100%", borderCollapse: "collapse" }}>
          <tbody>{complexityFields.map((f) => renderRow(f, assumptions.complexity?.[f]))}</tbody>
        </table>
      </div>
    </div>
  );
}

type WorkloadComplexity = {
  reasoning: ComplexityLevel;
  context: ContextComplexity;
  codingToolUse: ComplexityLevel;
  hallucinationSensitivity: ComplexityLevel;
};

const defaultWorkloadComplexity: WorkloadComplexity = {
  reasoning: "medium",
  context: "medium-context",
  codingToolUse: "medium",
  hallucinationSensitivity: "medium",
};

function levelScore(level: ComplexityLevel): number {
  return level === "low" ? 0 : level === "medium" ? 1 : 2;
}
function contextScore(context: ContextComplexity): number {
  return context === "short-context" ? 0 : context === "medium-context" ? 1 : 2;
}

function computeViableModelTiers(complexity: WorkloadComplexity): ViableModelTiers {
  const tiers: ViableModelTiers = { budget: true, premium: true, frontier: true };
  if (complexity.reasoning === "high") tiers.budget = false;
  if (complexity.context === "long-context") tiers.budget = false;
  if (complexity.hallucinationSensitivity === "high") tiers.budget = false;
  if (complexity.codingToolUse === "high") tiers.budget = false;
  return tiers;
}

function buildCapabilityConstraintMessages(complexity: WorkloadComplexity, viableModelTiers: ViableModelTiers): string[] {
  const messages: string[] = [];
  if (complexity.reasoning === "high") messages.push("High reasoning complexity excludes budget-tier models.");
  if (complexity.context === "long-context") messages.push("Long-context workloads prefer frontier-tier models.");
  if (complexity.codingToolUse === "high") messages.push("High coding/tool-use intensity favors GPT-5.5-class frontier models.");
  if (complexity.hallucinationSensitivity === "high") messages.push("High hallucination sensitivity limits selection to premium and frontier tiers.");
  if (!viableModelTiers.budget) messages.push("Budget-tier models are economically cheaper but excluded due to workload complexity requirements.");
  return [...new Set(messages)];
}

function selectViableApiModel(scenario: any, viableModelTiers: ViableModelTiers, complexity: WorkloadComplexity) {
  const candidates: { tier: ModelTier; model: any }[] = [];
  if (viableModelTiers.budget) candidates.push({ tier: "budget", model: scenario.reference_models.budget_api });
  if (viableModelTiers.premium) candidates.push({ tier: "premium", model: scenario.reference_models.premium_api });
  if (viableModelTiers.frontier) candidates.push({ tier: "frontier", model: scenario.reference_models.frontier_api });
  if (candidates.length === 0) candidates.push({ tier: "premium", model: scenario.reference_models.premium_api });
  if (complexity.codingToolUse === "high" && viableModelTiers.frontier) return { tier: "frontier" as const, model: scenario.reference_models.frontier_api };
  if (complexity.context === "long-context" && viableModelTiers.frontier) return { tier: "frontier" as const, model: scenario.reference_models.frontier_api };
  if (complexity.hallucinationSensitivity === "high") return [...candidates].filter((c) => c.tier !== "budget").sort((a, b) => a.model.monthly_cost - b.model.monthly_cost)[0];
  return [...candidates].sort((a, b) => a.model.monthly_cost - b.model.monthly_cost)[0];
}

function buildCapabilityAwareRecommendation(result: any, complexity: WorkloadComplexity, viableModelTiers: ViableModelTiers): CapabilityAwareRecommendation {
  const current = result.scenarios[0];
  const growth5x = result.scenarios[result.scenarios.length - 1];
  const selectedApi = selectViableApiModel(current, viableModelTiers, complexity);
  const selectedGrowthApi = selectViableApiModel(growth5x, viableModelTiers, complexity);
  const h100Current = current.self_hosted_h100;
  const h100Growth = growth5x.self_hosted_h100;
  const budgetCost = current.reference_models.budget_api.monthly_cost;
  const budgetExcludedDespiteCheaper = !viableModelTiers.budget && budgetCost < selectedApi.model.monthly_cost;
  const apiCheaper = selectedApi.model.monthly_cost < h100Current.self_hosted_monthly_cost;
  let recommendedOption: string;
  let rationale: string;
  if (apiCheaper) {
    recommendedOption = "API";
    rationale = `Among capability-viable API tiers, ${selectedApi.model.model_key} (${selectedApi.tier} tier) is recommended at $${selectedApi.model.monthly_cost}/month versus $${h100Current.self_hosted_monthly_cost}/month for self-hosted H100.`;
  } else {
    recommendedOption = "Self-hosted H100";
    rationale = `Self-hosted H100 is recommended at $${h100Current.self_hosted_monthly_cost}/month versus $${selectedApi.model.monthly_cost}/month for capability-viable ${selectedApi.model.model_key}.`;
  }
  if (budgetExcludedDespiteCheaper) rationale += " Budget-tier models are economically cheaper but excluded due to workload complexity requirements.";
  const growthApiCheaper = selectedGrowthApi.model.monthly_cost < h100Growth.self_hosted_monthly_cost;
  const migrationTrigger = growthApiCheaper
    ? "No migration trigger in modeled range. Capability-viable API inference remains economically favorable."
    : "At 5x growth, self-hosting becomes attractive among capability-viable options. Evaluate dedicated GPU infrastructure.";
  let preferredModelFamily: string | null = null;
  if (complexity.codingToolUse === "high") preferredModelFamily = "GPT-5.5";
  else if (complexity.hallucinationSensitivity === "high" || complexity.reasoning === "high") preferredModelFamily = "Claude Opus 4.7";
  else if (complexity.context === "long-context") preferredModelFamily = "Frontier-class (GPT-5.5)";
  return {
    recommendedOption, rationale, migrationTrigger,
    capabilityConstraints: buildCapabilityConstraintMessages(complexity, viableModelTiers),
    viableModelTiers,
    selectedApiTier: apiCheaper ? selectedApi.tier : null,
    preferredModelFamily,
  };
}

function analyzeWorkloadComplexity(complexity: WorkloadComplexity, viableModelTiers: ViableModelTiers) {
  const weightedScore =
    levelScore(complexity.reasoning) * 1.2 +
    contextScore(complexity.context) * 1.1 +
    levelScore(complexity.codingToolUse) * 1.3 +
    levelScore(complexity.hallucinationSensitivity);
  let recommendedTier: ModelTier;
  if (weightedScore <= 2.5 && viableModelTiers.budget) recommendedTier = "budget";
  else if (weightedScore <= 5.5 && viableModelTiers.premium) recommendedTier = "premium";
  else if (viableModelTiers.frontier) recommendedTier = "frontier";
  else if (viableModelTiers.premium) recommendedTier = "premium";
  else recommendedTier = "budget";
  if (!viableModelTiers[recommendedTier]) {
    recommendedTier = viableModelTiers.premium ? "premium" : viableModelTiers.frontier ? "frontier" : "budget";
  }
  const tierBadges = [
    { id: "budget" as const, label: "Budget viable", status: !viableModelTiers.budget ? "excluded" : recommendedTier === "budget" ? "recommended" : "viable" },
    { id: "premium" as const, label: "Premium recommended", status: !viableModelTiers.premium ? "excluded" : recommendedTier === "premium" ? "recommended" : viableModelTiers.budget ? "optional" : "viable" },
    { id: "frontier" as const, label: "Frontier required", status: !viableModelTiers.frontier ? "excluded" : recommendedTier === "frontier" ? "required" : "optional" },
  ];
  const emphasizeGpt = complexity.codingToolUse === "high" || complexity.context === "long-context" || complexity.reasoning === "high";
  const emphasizeClaude = complexity.reasoning === "high" || complexity.hallucinationSensitivity === "high" || (complexity.codingToolUse === "medium" && complexity.hallucinationSensitivity !== "low");
  return { recommendedTier, weightedScore, tierBadges, emphasizeGpt, emphasizeClaude, viableModelTiers };
}

function tierIsViable(tier: ModelTier | "h100", viableModelTiers: ViableModelTiers): boolean {
  if (tier === "h100") return true;
  return viableModelTiers[tier];
}

const REF_MODEL_BY_TIER = { budget: "budget_api", premium: "premium_api", frontier: "frontier_api" } as const;
const TIER_DISPLAY_LABEL = { budget: "Budget API", premium: "Premium API", frontier: "Frontier API" } as const;

function formatScenarioLabel(scenario: string): string {
  if (scenario === "current") return "current";
  if (scenario === "growth_2x") return "2x";
  if (scenario === "growth_5x") return "5x";
  return scenario;
}

type CrossoverTierAnalysis = {
  tier: ModelTier;
  tierLabel: string;
  modelKey: string;
  scenarioComparisons: { scenario: string; scenarioLabel: string; apiCost: number; h100Cost: number; h100Cheaper: boolean }[];
  h100CheaperAtLabels: string[];
  firstCrossoverLabel: string | null;
  noCrossoverInRange: boolean;
};

function buildCrossoverAnalysis(scenarios: any[], viableModelTiers: ViableModelTiers): CrossoverTierAnalysis[] {
  const tiersToAnalyze: ModelTier[] = [];
  if (viableModelTiers.premium) tiersToAnalyze.push("premium");
  if (viableModelTiers.frontier) tiersToAnalyze.push("frontier");
  return tiersToAnalyze.map((tier) => {
    const refKey = REF_MODEL_BY_TIER[tier];
    const scenarioComparisons = scenarios.map((scenario) => {
      const apiCost = scenario.reference_models[refKey].monthly_cost;
      const h100Cost = scenario.self_hosted_h100.self_hosted_monthly_cost;
      return { scenario: scenario.scenario, scenarioLabel: formatScenarioLabel(scenario.scenario), apiCost, h100Cost, h100Cheaper: h100Cost < apiCost };
    });
    const h100CheaperAtLabels = scenarioComparisons.filter((c) => c.h100Cheaper).map((c) => c.scenarioLabel);
    const firstCrossover = scenarioComparisons.find((c) => c.h100Cheaper);
    return {
      tier, tierLabel: TIER_DISPLAY_LABEL[tier],
      modelKey: scenarios[0].reference_models[refKey].model_key,
      scenarioComparisons, h100CheaperAtLabels,
      firstCrossoverLabel: firstCrossover?.scenarioLabel ?? null,
      noCrossoverInRange: firstCrossover === undefined,
    };
  });
}

type ManualWorkloadAssumptions = {
  monthly_queries: number | null;
  input_tokens_per_query: number | null;
  output_tokens_per_query: number | null;
  latency_sla_ms: number | null;
};

const emptyManualWorkload = (): ManualWorkloadAssumptions => ({
  monthly_queries: null, input_tokens_per_query: null,
  output_tokens_per_query: null, latency_sla_ms: null,
});

function buildSimulationDescription(plannerMessage: string, manual: ManualWorkloadAssumptions): string {
  const parts: string[] = [];
  if (manual.monthly_queries != null) parts.push(`${manual.monthly_queries} monthly queries`);
  if (manual.input_tokens_per_query != null) parts.push(`${manual.input_tokens_per_query} input tokens per query`);
  if (manual.output_tokens_per_query != null) parts.push(`${manual.output_tokens_per_query} output tokens per query`);
  if (manual.latency_sla_ms != null) parts.push(`${manual.latency_sla_ms}ms latency`);
  return parts.length > 0 ? parts.join(", ") : plannerMessage.trim();
}

/* ─────────────────────────── design tokens ─────────────────────────── */

const T = {
  bg: "#0F0F17",
  surface: "#1A1A24",
  border: "rgba(255,255,255,0.09)",
  borderStrong: "rgba(255,255,255,0.16)",
  text: "#F9FAFB",
  textMuted: "#C7CCD4",
  textDim: "#8B92A0",
  ink: "#0A0A12",
  amber: "#F59E0B",
  amberDim: "rgba(245,158,11,0.12)",
  amberBorder: "rgba(245,158,11,0.25)",
  violet: "#A78BFA",
  violetDim: "rgba(167,139,250,0.14)",
  violetBorder: "rgba(167,139,250,0.3)",
  // Darker shade for solid-fill buttons/badges where white text sits on top —
  // T.violet itself is now too light for white text to stay readable on.
  violetSolid: "#7C3AED",
  pink: "#EC4899",
  pinkDim: "rgba(236,72,153,0.12)",
  pinkBorder: "rgba(236,72,153,0.25)",
  blue: "#3B82F6",
  blueDim: "rgba(59,130,246,0.12)",
  blueBorder: "rgba(59,130,246,0.25)",
  mono: "'DM Mono', 'Fira Code', monospace",
  sans: "'Syne', 'DM Sans', system-ui, sans-serif",
  // Type scale — bumped up twice after feedback that the UI read as too small
  // and the gray/violet-on-black combination was hard to read. Keep using
  // these tokens for new text rather than raw numbers.
  fontMicro: 13,   // badges, tags, role labels ("YOU", "PROVIDED", section index numbers)
  fontSmall: 15,   // field labels, captions, secondary metadata
  fontBody: 17,    // primary readable content — chat messages, table values, inputs
  fontHeading: 19, // card titles, panel headers
};

/* ─────────────────── reusable components ─────────────────── */

function SectionHeader({ index, label }: { index: string; label: string }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 20 }}>
      <span style={{ fontFamily: T.mono, fontSize: T.fontMicro, letterSpacing: "0.22em", color: T.textDim }}>{index}</span>
      <div style={{ flex: 1, height: 1, background: T.border }} />
      <span style={{ fontFamily: T.mono, fontSize: T.fontMicro, letterSpacing: "0.22em", color: T.textMuted, textTransform: "uppercase" }}>{label}</span>
    </div>
  );
}

function Card({ children, accent, style }: { children: React.ReactNode; accent?: string; style?: React.CSSProperties }) {
  return (
    <div style={{
      background: T.surface,
      border: `1px solid ${T.border}`,
      borderRadius: 10,
      padding: "18px 20px",
      position: "relative",
      overflow: "hidden",
      ...(accent ? { borderLeft: `2px solid ${accent}` } : {}),
      ...style,
    }}>
      {children}
    </div>
  );
}

function SegmentedControl({ options, value, onChange, accent = T.violet }: {
  options: { label: string; value: string }[];
  value: string;
  onChange: (v: string) => void;
  accent?: string;
}) {
  return (
    <div style={{ display: "flex", background: T.bg, border: `1px solid ${T.border}`, borderRadius: 7, padding: 2, gap: 2 }}>
      {options.map((opt) => {
        const active = opt.value === value;
        return (
          <button key={opt.value} type="button" onClick={() => onChange(opt.value)}
            style={{
              flex: 1, padding: "6px 10px", fontSize: 11, fontFamily: T.mono,
              border: active ? `1px solid ${accent}30` : "1px solid transparent",
              borderRadius: 5, background: active ? `${accent}18` : "transparent",
              color: active ? accent : T.textMuted, cursor: "pointer",
              transition: "all 0.15s", letterSpacing: "0.04em",
            }}>
            {opt.label}
          </button>
        );
      })}
    </div>
  );
}

function StatusPill({ tone, label }: { tone: "collecting" | "ready" | "estimated" | "needs_user"; label: string }) {
  const colors = {
    collecting: { bg: "rgba(55,65,81,0.4)", border: T.border, text: T.textMuted },
    ready: { bg: T.violetDim, border: T.violetBorder, text: "#A78BFA" },
    estimated: { bg: T.amberDim, border: T.amberBorder, text: "#FCD34D" },
    needs_user: { bg: T.amberDim, border: T.amberBorder, text: "#FCD34D" },
  };
  const c = colors[tone];
  return (
    <div style={{ display: "inline-flex", alignItems: "center", gap: 6, background: c.bg, border: `1px solid ${c.border}`, borderRadius: 20, padding: "4px 12px" }}>
      <div style={{ width: 5, height: 5, borderRadius: "50%", background: c.text, boxShadow: tone !== "collecting" ? `0 0 6px ${c.text}` : "none" }} />
      <span style={{ fontSize: T.fontMicro, fontFamily: T.mono, letterSpacing: "0.1em", color: c.text }}>{label}</span>
    </div>
  );
}

function SliderInput({ id, label, value, min, max, unit = "%", onChange }: {
  id: string; label: string; value: number; min: number; max: number; unit?: string; onChange: (v: number) => void;
}) {
  const pct = ((value - min) / (max - min)) * 100;
  return (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 12 }}>
        <label htmlFor={id} style={{ fontSize: 12, color: T.text, fontFamily: T.sans, fontWeight: 500 }}>{label}</label>
        <span style={{ fontFamily: T.mono, fontSize: 13, color: T.amber, minWidth: 44, textAlign: "right" }}>{value}{unit}</span>
      </div>
      <div style={{ position: "relative", height: 4, background: T.border, borderRadius: 2 }}>
        <div style={{ position: "absolute", left: 0, width: `${pct}%`, height: "100%", background: `linear-gradient(90deg, ${T.violet}, ${T.amber})`, borderRadius: 2, transition: "width 0.1s" }} />
        <input id={id} type="range" min={min} max={max} value={value} onChange={(e) => onChange(Number(e.target.value))}
          style={{ position: "absolute", top: -7, left: 0, width: "100%", height: 18, opacity: 0, cursor: "pointer", zIndex: 2 }} />
        <div style={{
          position: "absolute", top: "50%", left: `calc(${pct}% - 8px)`, transform: "translateY(-50%)",
          width: 14, height: 14, borderRadius: "50%", background: T.amber, border: `2px solid ${T.bg}`,
          boxShadow: `0 0 8px ${T.amber}60`, pointerEvents: "none", transition: "left 0.1s",
        }} />
      </div>
    </div>
  );
}

function InlineTag({ label, color }: { label: string; color: "violet" | "amber" | "pink" | "blue" | "dim" }) {
  const map = {
    violet: { bg: T.violetDim, border: T.violetBorder, text: "#A78BFA" },
    amber: { bg: T.amberDim, border: T.amberBorder, text: "#FCD34D" },
    pink: { bg: T.pinkDim, border: T.pinkBorder, text: "#F472B6" },
    blue: { bg: T.blueDim, border: T.blueBorder, text: "#93C5FD" },
    dim: { bg: "rgba(55,65,81,0.3)", border: T.border, text: T.textMuted },
  };
  const c = map[color];
  return (
    <span style={{ background: c.bg, border: `1px solid ${c.border}`, borderRadius: 4, padding: "2px 8px", fontSize: 10, fontFamily: T.mono, color: c.text, letterSpacing: "0.08em" }}>
      {label}
    </span>
  );
}

/* ─────────────────────────── main component ─────────────────────────── */

export default function Home() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [chatInput, setChatInput] = useState("");
  const [manualWorkload, setManualWorkload] = useState<ManualWorkloadAssumptions>(emptyManualWorkload);
  const [plannerLoading, setPlannerLoading] = useState(false);
  const [plannerError, setPlannerError] = useState<string | null>(null);
  const [planResult, setPlanResult] = useState<PlanResponse | null>(null);
  const [plannerConversation, setPlannerConversation] = useState<PlannerTurn[]>([]);
  const [adkClarificationPending, setAdkClarificationPending] = useState(false);
  const [adkOriginalMessage, setAdkOriginalMessage] = useState<string>("");
  const [adkLastVerdict, setAdkLastVerdict] = useState<string | null>(null);
  const [thinkingPhraseIndex, setThinkingPhraseIndex] = useState(0);
  const [assumptions, setAssumptions] = useState<InfrastructureAssumptions>(defaultAssumptions);
  const [deploymentPreset, setDeploymentPreset] = useState<DeploymentPresetId | null>(null);
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [manualOpen, setManualOpen] = useState(false);
  const [specEditMode, setSpecEditMode] = useState(false);
  const [specDraft, setSpecDraft] = useState<Record<string, any>>({});
  const [recalcLoading, setRecalcLoading] = useState(false);

  // Rotating "thinking" phrases shown while waiting on the ADK pipeline.
  // Generic on purpose — the backend doesn't stream per-agent progress, so a
  // specific stage label (e.g. "Researching pricing...") could be wrong if the
  // request gets gated early. These phrases just keep the wait feeling alive
  // and honest, without claiming to know exactly what's running right now.
  const THINKING_PHRASES = [
    "Thinking...",
    "Reading through your workload...",
    "Gathering thoughts...",
    "Weighing the options...",
    "Crunching the numbers...",
    "Almost there...",
  ];
  useEffect(() => {
    if (!plannerLoading) {
      setThinkingPhraseIndex(0);
      return;
    }
    const interval = setInterval(() => {
      setThinkingPhraseIndex((prev) => (prev + 1) % THINKING_PHRASES.length);
    }, 2800);
    return () => clearInterval(interval);
  }, [plannerLoading]);

  function applyDeploymentPreset(presetId: DeploymentPresetId) {
    setAssumptions(DEPLOYMENT_PRESETS[presetId].values);
    setDeploymentPreset(presetId);
  }
  function updateAssumptions(patch: Partial<InfrastructureAssumptions>) {
    setDeploymentPreset(null);
    setAssumptions((prev) => ({ ...prev, ...patch }));
  }

  function syncPlanToLocalState(data: PlanResponse) {
    const workload = data.structured_assumptions.workload;
    const operational = data.structured_assumptions.operational;
    if (workload) {
      setManualWorkload({
        monthly_queries: getFieldInternal(workload.monthly_queries) as number | null,
        input_tokens_per_query: getFieldInternal(workload.input_tokens_per_query) as number | null,
        output_tokens_per_query: getFieldInternal(workload.output_tokens_per_query) as number | null,
        latency_sla_ms: getFieldInternal(workload.latency_sla_ms) as number | null,
      });
    }
    if (operational) {
      setAssumptions({
        gpuUtilizationPct: operational.gpu_utilization_pct,
        enterpriseApiDiscountPct: operational.enterprise_api_discount_pct,
        burstiness: operational.burstiness_factor as BurstinessLevel,
        reservedFailoverCapacityPct: operational.failover_reserve_pct,
      });
    }
  }

  async function sendChatMessage(message: string) {
    const trimmed = message.trim();
    if (!trimmed || plannerLoading) return;

    const userTurn: PlannerTurn = { id: `user-${Date.now()}`, role: "user", content: trimmed };
    setPlannerConversation((prev) => [...prev, userTurn]);
    setChatInput("");

    // If ADK Judge asked a clarifying question, route this answer back into /adk/simulate
    if (adkClarificationPending) {
      try {
        setPlannerLoading(true);
        setPlannerError(null);
        const combined = `${adkOriginalMessage}\n\nAdditional context: ${trimmed}`;
        const response = await fetch(`${API_BASE_URL}/adk/simulate`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message: combined, user_id: "frontend_user" }),
        });
        const adkData = await response.json();
        setAdkLastVerdict(adkData.verdict ?? null);

        if (adkData.verdict === "infeasible") {
          // Terminal explanation, not a question — close the clarification loop.
          setAdkClarificationPending(false);
          setAdkOriginalMessage("");
          setPlannerConversation((prev) => [...prev, {
            id: `planner-infeasible-${Date.now()}`,
            role: "planner",
            content: adkData.clarifying_question || "That configuration isn't possible — closed models can't be self-hosted.",
          }]);
          return;
        }

        if (adkData.verdict === "needs_user") {
          // Still not enough info — ask again
          setPlannerConversation((prev) => [...prev, {
            id: `planner-clarify-${Date.now()}`,
            role: "planner",
            content: adkData.clarifying_question || "Could you provide more details?",
          }]);
          setAdkOriginalMessage(combined); // accumulate context for next round
          return;
        }

        // Got a valid result — clear clarification state and render
        setAdkClarificationPending(false);
        setAdkOriginalMessage("");
        setPlanResult({
          assistant_message: "",
          structured_assumptions: adkSpecToStructuredAssumptions(adkData.workload_spec),
          missing_fields: [],
          clarification_questions: [],
          assumed_fields: Object.keys(adkData.workload_spec?.field_confidence ?? {}),
          ready_to_simulate: true,
        });
        const rec = adkData.final_recommendation;
        setResult({
          scenarios: adkData.cost_scenarios ?? [],
          recommendation: rec ? {
            recommendation: rec.recommendation,
            rationale: rec.recommendation_rationale,
            crossover_note: rec.breakeven_monthly_queries
              ? `Breakeven at approx. ${rec.breakeven_monthly_queries.toLocaleString()} queries/month`
              : "No crossover in the modeled range.",
            confidence_score: rec.confidence_score,
            confidence_explanation: rec.confidence_explanation,
            latency_flag: rec.latency_flag ?? "none",
            latency_note: rec.latency_note ?? "",
            quality_gap_warning: rec.quality_gap_warning ?? "",
            toolchain_friction: rec.toolchain_friction ?? "none",
            toolchain_friction_note: rec.toolchain_friction_note ?? "",
          } : null,
          _adkVerdict: adkData.verdict,
          _adkWorkloadSpec: adkData.workload_spec,
        });
      } catch (error) {
        setPlannerError(error instanceof Error ? error.message : "Unable to reach the simulator.");
      } finally {
        setPlannerLoading(false);
      }
      return;
    }

    // Route directly to the ADK full pipeline — ParseJudge → PricingAgent → CostEngine → Reasoning
    try {
      setPlannerLoading(true);
      setPlannerError(null);
      const response = await fetch(`${API_BASE_URL}/adk/simulate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: trimmed, user_id: "frontend_user" }),
      });
      if (!response.ok) throw new Error(`ADK pipeline failed (${response.status})`);
      const adkData = await response.json();
      setAdkLastVerdict(adkData.verdict ?? null);

      if (adkData.verdict === "infeasible") {
        // Terminal explanation, not a question — never open the clarification loop.
        setPlannerConversation((prev) => [...prev, {
          id: `planner-infeasible-${Date.now()}`,
          role: "planner",
          content: adkData.clarifying_question || "That configuration isn't possible — closed models can't be self-hosted.",
        }]);
        return;
      }

      if (adkData.verdict === "needs_user") {
        setAdkClarificationPending(true);
        setAdkOriginalMessage(trimmed);
        setPlannerConversation((prev) => [...prev, {
          id: `planner-clarify-${Date.now()}`,
          role: "planner",
          content: adkData.clarifying_question || "Could you provide more details about your workload?",
        }]);
        return;
      }

      // verdict === "pass" — results ready
      setAdkClarificationPending(false);
      setAdkOriginalMessage("");
      setPlanResult({
        assistant_message: "",
        structured_assumptions: adkSpecToStructuredAssumptions(adkData.workload_spec),
        missing_fields: [],
        clarification_questions: [],
        assumed_fields: Object.keys(adkData.workload_spec?.field_confidence ?? {}),
        ready_to_simulate: true,
      });
      const spec = adkData.workload_spec;
      const parts: string[] = [];
      if (spec?.monthly_queries) parts.push(`${Number(spec.monthly_queries).toLocaleString()} queries/mo`);
      if (spec?.input_tokens_per_query) parts.push(`${spec.input_tokens_per_query} input tokens`);
      if (spec?.output_tokens_per_query) parts.push(`${spec.output_tokens_per_query} output tokens`);
      if (spec?.latency_sla) parts.push(`${spec.latency_sla} latency`);
      setPlannerConversation((prev) => [...prev, {
        id: `planner-${Date.now()}`,
        role: "planner",
        content: parts.length > 0
          ? `Analyzed: ${parts.join(" · ")}. Full cost breakdown below.`
          : "Workload analyzed. Full cost breakdown below.",
      }]);

      const rec = adkData.final_recommendation;
      setResult({
        scenarios: adkData.cost_scenarios ?? [],
        recommendation: rec ? {
          recommendation: rec.recommendation,
          rationale: rec.recommendation_rationale,
          crossover_note: rec.breakeven_monthly_queries
            ? `Breakeven at approx. ${rec.breakeven_monthly_queries.toLocaleString()} queries/month`
            : "No crossover in the modeled range.",
          confidence_score: rec.confidence_score,
          confidence_explanation: rec.confidence_explanation,
          latency_flag: rec.latency_flag ?? "none",
          latency_note: rec.latency_note ?? "",
          quality_gap_warning: rec.quality_gap_warning ?? "",
          toolchain_friction: rec.toolchain_friction ?? "none",
          toolchain_friction_note: rec.toolchain_friction_note ?? "",
        } : null,
        _adkVerdict: adkData.verdict,
        _adkWorkloadSpec: adkData.workload_spec,
      });
    } catch (error) {
      setPlannerError(error instanceof Error ? error.message : "Unable to reach the ADK pipeline. Is the server running?");
    } finally {
      setPlannerLoading(false);
    }
  }

  const plannerStatus = getVerdictStatus(adkLastVerdict, plannerLoading);
  // Button shows whenever the user has typed something — ADK handles parsing inline.
  const canRunSimulation = chatInput.trim().length > 0;

  const runSimulation = useCallback(async () => {
    if (!canRunSimulation) return;
    try {
      setLoading(true);
      const firstUserMessage = plannerConversation.find((turn) => turn.role === "user")?.content ?? "";
      const message = buildSimulationDescription(firstUserMessage, manualWorkload);
      const response = await fetch(`${API_BASE_URL}/adk/simulate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message, user_id: "frontend_user" }),
      });
      const adkData = await response.json();

      // If ADK needs clarification, surface the question and arm the clarification loop
      if (adkData.verdict === "needs_user") {
        setAdkClarificationPending(true);
        setAdkOriginalMessage(message);
        setPlannerConversation((prev) => [
          ...prev,
          {
            id: `planner-clarify-${Date.now()}`,
            role: "planner",
            content: adkData.clarifying_question || "Could you provide more details about your workload?",
          },
        ]);
        return;
      }

      // Clear any stale clarification state on successful run
      setAdkClarificationPending(false);
      setAdkOriginalMessage("");

      // Adapt ADK response → shape the legacy rendering logic expects
      const rec = adkData.final_recommendation;
      setResult({
        scenarios: adkData.cost_scenarios ?? [],
        recommendation: rec
          ? {
              recommendation: rec.recommendation,
              rationale: rec.recommendation_rationale,
              crossover_note: rec.breakeven_monthly_queries
                ? `Breakeven at approx. ${rec.breakeven_monthly_queries.toLocaleString()} queries/month`
                : "No crossover in the modeled range.",
              confidence_score: rec.confidence_score,
              confidence_explanation: rec.confidence_explanation,
              latency_flag: rec.latency_flag ?? "none",
              latency_note: rec.latency_note ?? "",
              quality_gap_warning: rec.quality_gap_warning ?? "",
              toolchain_friction: rec.toolchain_friction ?? "none",
              toolchain_friction_note: rec.toolchain_friction_note ?? "",
            }
          : null,
        _adkVerdict: adkData.verdict,
        _adkWorkloadSpec: adkData.workload_spec,
      });
    } catch (error) {
      console.error(error);
    } finally {
      setLoading(false);
    }
  }, [canRunSimulation, manualWorkload, plannerConversation]);

  const assumptionsSnapshot = useRef(JSON.stringify(assumptions));
  useEffect(() => {
    // ADK pipeline doesn't accept real-time assumption overrides; reset snapshot to prevent stale re-runs
    assumptionsSnapshot.current = JSON.stringify(assumptions);
  }, [assumptions]);

  const examples = [
    "Customer support chatbot for a seed-stage SaaS startup",
    "AI research copilot for a growing B2B platform",
    "Enterprise customer support chatbot for 10,000 users",
  ];

  // ── FIX: loadExample uses a stable index to avoid hydration mismatch ──────
  const [exampleIndex] = useState(() => Math.floor(Math.random() * examples.length));
  function loadExample() {
    setChatInput(examples[exampleIndex]);
    setPlanResult(null);
    setPlannerConversation([]);
    setManualWorkload(emptyManualWorkload());
    setResult(null);
  }

  function handleChatKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); void sendChatMessage(chatInput); }
  }

  const costChartSeries = [
    { dataKey: "budgetApi",   label: "Budget API (cheapest)",  color: "#8B5CF6" },
    { dataKey: "premiumApi",  label: "Premium API",            color: "#3B82F6" },
    { dataKey: "frontierApi", label: "Frontier API",           color: "#EC4899" },
    { dataKey: "coreweave",   label: "CoreWeave H100",         color: "#F59E0B" },
    { dataKey: "lambda",      label: "Lambda Labs H100",       color: "#10B981" },
    { dataKey: "trainium",    label: "AWS Trainium2",          color: "#F97316" },
    { dataKey: "tpu",         label: "GCP TPU v5e",            color: "#06B6D4" },
  ] as const;

  const costChartLabelByKey = Object.fromEntries(costChartSeries.map((s) => [s.dataKey, s.label])) as Record<string, string>;

  const costChartData = result?.scenarios?.map((scenario: any) => {
    const gpuMap = Object.fromEntries(
      (scenario.gpu_providers ?? []).map((g: any) => [g.provider_key, g.monthly_cost])
    );
    return {
      scenario: scenario.scenario,
      budgetApi:   scenario.reference_models?.budget_api?.monthly_cost ?? null,
      premiumApi:  scenario.reference_models?.premium_api?.monthly_cost ?? null,
      frontierApi: scenario.reference_models?.frontier_api?.monthly_cost ?? null,
      coreweave:   gpuMap["coreweave_h100"] ?? null,
      lambda:      gpuMap["lambda_labs_h100"] ?? null,
      trainium:    gpuMap["aws_trainium2"] ?? null,
      tpu:         gpuMap["gcp_tpu_v5"] ?? null,
    };
  }) ?? [];

  const fmtCost = (v: number) => {
    if (v < 10) return `$${v.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
    return `$${Math.round(v).toLocaleString()}`;
  };
  const assumptionImpact = getAssumptionImpactSummary(assumptions);
  // Derive recommendation from the backend-provided recommendation field
  const backendRec = result?.recommendation ?? null;
  const recStr = String(backendRec?.recommendation ?? "").toLowerCase();
  const isApiRecommended = recStr.includes("api") && !recStr.includes("open");
  const isOpenWeightRecommended = recStr.includes("open") || recStr.includes("open-weight");
  const verdictColor = isApiRecommended ? "#A78BFA" : isOpenWeightRecommended ? "#34D399" : T.amber;
  const headerBg = isApiRecommended ? T.violet : isOpenWeightRecommended ? "#065F46" : T.amber;
  const headerTextColor = isApiRecommended ? "#E9D5FF" : isOpenWeightRecommended ? "#D1FAE5" : T.ink;
  const borderColor = isApiRecommended ? T.violetBorder : isOpenWeightRecommended ? "#064E3B" : T.amberBorder;

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@400;600;700;800&display=swap');
        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
        html { background: ${T.bg}; }
        body { background: ${T.bg}; font-family: ${T.sans}; color: ${T.text}; }
        ::selection { background: ${T.violetDim}; color: #C4B5FD; }
        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: ${T.border}; border-radius: 2px; }
        .sim-btn { transition: all 0.15s; }
        .sim-btn:hover:not(:disabled) { transform: translateY(-1px); box-shadow: 0 4px 20px rgba(124,58,237,0.35); }
        .sim-btn:active:not(:disabled) { transform: translateY(0); }
        .collapse-btn { background: none; border: none; cursor: pointer; display: flex; align-items: center; justify-content: space-between; width: 100%; padding: 0; }
        .preset-btn { transition: all 0.15s; }
        .preset-btn:hover { border-color: rgba(124,58,237,0.4) !important; }
        input[type=number] { -moz-appearance: textfield; }
        input[type=number]::-webkit-outer-spin-button,
        input[type=number]::-webkit-inner-spin-button { -webkit-appearance: none; margin: 0; }
      `}</style>

      <main style={{ minHeight: "100vh", background: T.bg, color: T.text }}>

        {/* ── NAV ── */}
        <nav style={{ borderBottom: `1px solid ${T.border}`, padding: "14px 40px", display: "flex", alignItems: "center", justifyContent: "space-between", position: "sticky", top: 0, zIndex: 50, background: `${T.bg}E8`, backdropFilter: "blur(12px)" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
            <div style={{ width: 28, height: 28, borderRadius: 6, background: `linear-gradient(135deg, ${T.violet}, ${T.amber})`, display: "flex", alignItems: "center", justifyContent: "center" }}>
              <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
                <path d="M2 7L7 2L12 7L7 12L2 7Z" stroke="white" strokeWidth="1.5" strokeLinejoin="round" />
              </svg>
            </div>
            <span style={{ fontFamily: T.sans, fontSize: 14, fontWeight: 700, letterSpacing: "-0.02em", color: T.text }}>AI Cost Simulator</span>
          </div>
        </nav>

        {/* ── HERO ── */}
        <section style={{ maxWidth: 1100, margin: "0 auto", padding: "72px 40px 60px" }}>
          <div style={{ display: "inline-flex", alignItems: "center", gap: 8, border: `1px solid ${T.violetBorder}`, borderRadius: 20, padding: "5px 14px", marginBottom: 32, background: T.violetDim }}>
            <div style={{ width: 4, height: 4, borderRadius: "50%", background: T.violet }} />
            <span style={{ fontFamily: T.mono, fontSize: 9, letterSpacing: "0.22em", color: "#A78BFA" }}>AI INFRA + FOUNDATION MODEL ECONOMICS</span>
          </div>
          <h1 style={{ fontFamily: T.sans, fontSize: "clamp(42px, 6vw, 72px)", fontWeight: 800, lineHeight: 0.95, letterSpacing: "-0.03em", marginBottom: 28 }}>
            AI Workload Cost<br />
            <em style={{ fontStyle: "italic", color: T.amber }}>Simulated</em>{" "}Across API<br />
            <span style={{ color: T.textMuted }}>and H100 Infra</span>
          </h1>
          <p style={{ fontSize: 16, lineHeight: 1.7, color: T.textMuted, maxWidth: 560 }}>
            Describe your workload in plain English. The planner clarifies missing details, then the deterministic simulator models growth scenarios and compares API inference against self-hosted H100 capacity.
          </p>
        </section>

        <div style={{ maxWidth: 1100, margin: "0 auto", padding: "0 40px 100px" }}>

          {/* ══ 01 · WORKLOAD PLANNER ══ */}
          <SectionHeader index="01" label="Workload Planner" />

          <div style={{ border: `1px solid ${T.border}`, borderRadius: 12, background: T.surface, overflow: "hidden" }}>
            {/* header */}
            <div style={{ padding: "20px 24px", borderBottom: `1px solid ${T.border}`, display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: 12 }}>
              <div>
                <div style={{ fontSize: T.fontHeading, fontWeight: 600, color: T.text, fontFamily: T.sans, marginBottom: 4 }}>Workload Planner</div>
                <div style={{ fontSize: T.fontSmall, color: T.textMuted, fontFamily: T.sans }}>Describe your AI workload in plain English. The planner collects assumptions conversationally and enables simulation when ready.</div>
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                <StatusPill tone={plannerStatus.tone} label={plannerStatus.label} />
                {/* ── FIX: suppressHydrationWarning stops browser-extension attr mismatch ── */}
                <button
                  suppressHydrationWarning
                  type="button"
                  onClick={loadExample}
                  style={{ fontFamily: T.mono, fontSize: T.fontMicro, letterSpacing: "0.08em", padding: "6px 14px", border: `1px solid ${T.borderStrong}`, borderRadius: 6, background: "transparent", color: T.textMuted, cursor: "pointer" }}
                >
                  LOAD EXAMPLE
                </button>
              </div>
            </div>

            {/* body */}
            <div style={{ display: "grid", gridTemplateColumns: "1fr 300px", gap: 0 }}>
              {/* chat column */}
              <div style={{ padding: "20px 24px", borderRight: `1px solid ${T.border}` }}>
                {(plannerConversation.length > 0 || plannerLoading) && (
                  <div style={{ marginBottom: 16, maxHeight: 300, overflowY: "auto", display: "flex", flexDirection: "column", gap: 10 }}>
                    {plannerConversation.map((turn) => (
                      <div key={turn.id} style={{
                        padding: "12px 16px", borderRadius: 8,
                        border: `1px solid ${turn.role === "user" ? T.border : T.violetBorder}`,
                        background: turn.role === "user" ? T.bg : T.violetDim,
                        marginLeft: turn.role === "user" ? 32 : 0,
                        marginRight: turn.role === "planner" ? 32 : 0,
                      }}>
                        <div style={{ fontFamily: T.mono, fontSize: T.fontMicro, letterSpacing: "0.14em", color: turn.role === "user" ? T.textDim : "#7C3AED", marginBottom: 6 }}>
                          {turn.role === "user" ? "YOU" : "PLANNER"}
                        </div>
                        <div style={{ fontSize: T.fontBody, lineHeight: 1.65, color: "#D1D5DB", whiteSpace: "pre-wrap", fontFamily: T.sans }}>{turn.content}</div>
                      </div>
                    ))}
                    {plannerLoading && (
                      <div style={{ display: "flex", gap: 8, padding: "12px 16px", alignItems: "center" }}>
                        <div style={{ display: "flex", gap: 4 }}>
                          {[0, 1, 2].map((i) => (
                            <div key={i} style={{ width: 6, height: 6, borderRadius: "50%", background: T.violet, animation: `pulse 1.2s ${i * 0.2}s infinite`, opacity: 0.7 }} />
                          ))}
                        </div>
                        <span style={{ fontFamily: T.sans, fontSize: T.fontSmall, color: T.textMuted, fontStyle: "italic" }}>
                          {THINKING_PHRASES[thinkingPhraseIndex]}
                        </span>
                        <style>{`@keyframes pulse { 0%,100%{opacity:0.3;transform:scale(0.8)} 50%{opacity:1;transform:scale(1)} }`}</style>
                      </div>
                    )}
                  </div>
                )}

                {plannerError && (
                  <div style={{ padding: "10px 14px", borderRadius: 7, border: `1px solid ${T.pinkBorder}`, background: T.pinkDim, fontSize: T.fontBody, color: "#F9A8D4", fontFamily: T.sans, marginBottom: 12 }}>
                    {plannerError}
                  </div>
                )}

                {/* ── FIX: suppressHydrationWarning on textarea ── */}
                <textarea
                  suppressHydrationWarning
                  value={chatInput}
                  onChange={(e) => setChatInput(e.target.value)}
                  onKeyDown={handleChatKeyDown}
                  disabled={plannerLoading}
                  placeholder={adkClarificationPending
                    ? "Answer the question above to complete your simulation…"
                    : "e.g. Customer support chatbot serving 50k monthly users with medium context windows…"}
                  style={{
                    width: "100%", height: 96, resize: "none",
                    background: T.bg, border: `1px solid ${adkClarificationPending ? "#78350F" : T.borderStrong}`,
                    borderRadius: 8, padding: "12px 14px", fontSize: T.fontBody,
                    fontFamily: T.sans, color: T.text, outline: "none", lineHeight: 1.6,
                  }}
                />
                <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginTop: 8 }}>
                  <span style={{ fontFamily: T.mono, fontSize: T.fontMicro, color: T.textDim, letterSpacing: "0.1em" }}>↵ SEND · SHIFT+↵ NEWLINE</span>
                  {canRunSimulation && (
                    <button
                      type="button"
                      onClick={() => sendChatMessage(chatInput)}
                      disabled={plannerLoading}
                      className="sim-btn"
                      style={{
                        padding: "9px 22px", background: plannerLoading ? T.violetDim : T.violetSolid,
                        border: `1px solid ${plannerLoading ? T.violetBorder : T.violetSolid}`, borderRadius: 7,
                        fontSize: T.fontSmall, fontFamily: T.mono, letterSpacing: "0.1em",
                        color: plannerLoading ? "#A78BFA" : "#fff", cursor: plannerLoading ? "not-allowed" : "pointer", fontWeight: 600,
                      }}
                    >
                      {plannerLoading ? "ANALYZING…" : "ANALYZE →"}
                    </button>
                  )}
                </div>
              </div>

              {/* assumptions panel */}
              <div style={{ padding: "20px", background: T.bg }}>
                <div style={{ fontFamily: T.mono, fontSize: T.fontMicro, letterSpacing: "0.18em", color: "#7C3AED", marginBottom: 4 }}>STRUCTURED ASSUMPTIONS</div>
                {planResult
                  ? <AssumptionsTable assumptions={planResult.structured_assumptions} />
                  : <p style={{ fontSize: T.fontSmall, color: T.textDim, fontFamily: T.sans, marginTop: 16, lineHeight: 1.6 }}>Assumptions appear here as you chat with the planner.</p>
                }
              </div>
            </div>
          </div>

          {/* ══ 03 · SIMULATION RESULTS ══ */}
          {result && (
            <>
              <div style={{ marginTop: 48, display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                <SectionHeader index="03" label="Simulation Results" />
                <span style={{ fontFamily: T.mono, fontSize: T.fontMicro, letterSpacing: "0.14em", color: "#7C3AED", background: "rgba(124,58,237,0.1)", border: "1px solid rgba(124,58,237,0.25)", borderRadius: 12, padding: "3px 10px", marginBottom: 20 }}>
                  POWERED BY GOOGLE ADK
                </span>
              </div>

              {/* workload spec provenance strip — editable */}
              {result._adkWorkloadSpec && (() => {
                const spec = result._adkWorkloadSpec as any;
                const draft = specEditMode ? specDraft : spec;
                const fmtNum = (n: number) => n >= 1_000_000 ? `${(n/1_000_000).toFixed(1)}M` : n >= 1_000 ? `${(n/1_000).toFixed(0)}K` : String(n);
                const inputStyle = { background: T.bg, border: `1px solid ${T.violetBorder}`, borderRadius: 4, padding: "2px 6px", fontFamily: T.mono, fontSize: T.fontSmall, fontWeight: 700, color: T.text, outline: "none", width: 100 };
                const selectStyle = { ...inputStyle, width: 120, cursor: "pointer" };

                async function handleRecalculate() {
                  setRecalcLoading(true);
                  try {
                    const response = await fetch(`${API_BASE_URL}/adk/recalculate`, {
                      method: "POST",
                      headers: { "Content-Type": "application/json" },
                      body: JSON.stringify({ workload_spec: specDraft, user_id: "frontend_user" }),
                    });
                    const adkData = await response.json();
                    const rec = adkData.final_recommendation;
                    setResult((prev: any) => ({
                      ...prev,
                      scenarios: adkData.cost_scenarios ?? prev.scenarios,
                      recommendation: rec ? {
                        recommendation: rec.recommendation,
                        rationale: rec.recommendation_rationale,
                        crossover_note: rec.breakeven_monthly_queries
                          ? `Breakeven at approx. ${rec.breakeven_monthly_queries.toLocaleString()} queries/month`
                          : "No crossover in the modeled range.",
                        confidence_score: rec.confidence_score,
                        confidence_explanation: rec.confidence_explanation,
                        latency_flag: rec.latency_flag ?? "none",
                        latency_note: rec.latency_note ?? "",
                        quality_gap_warning: rec.quality_gap_warning ?? "",
                        toolchain_friction: rec.toolchain_friction ?? "none",
                        toolchain_friction_note: rec.toolchain_friction_note ?? "",
                      } : prev.recommendation,
                      _adkWorkloadSpec: specDraft,
                    }));
                    setSpecEditMode(false);
                  } catch (e) {
                    console.error(e);
                  } finally {
                    setRecalcLoading(false);
                  }
                }

                return (
                  <div style={{ background: T.surface, border: `1px solid ${specEditMode ? T.violetBorder : T.border}`, borderRadius: 10, padding: "14px 18px", marginBottom: 14 }}>
                    <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 12 }}>
                      <span style={{ fontFamily: T.mono, fontSize: T.fontMicro, letterSpacing: "0.16em", color: T.textMuted }}>
                        PARSED BY ADK · PARSING AGENT EXTRACTED
                      </span>
                      <div style={{ display: "flex", gap: 8 }}>
                        {specEditMode ? (
                          <>
                            <button onClick={() => setSpecEditMode(false)} style={{ fontFamily: T.mono, fontSize: T.fontMicro, letterSpacing: "0.1em", color: T.textMuted, background: "none", border: `1px solid ${T.border}`, borderRadius: 4, padding: "3px 10px", cursor: "pointer" }}>
                              CANCEL
                            </button>
                            <button onClick={handleRecalculate} disabled={recalcLoading} style={{ fontFamily: T.mono, fontSize: T.fontMicro, letterSpacing: "0.1em", color: "#fff", background: T.violetSolid, border: "none", borderRadius: 4, padding: "3px 12px", cursor: "pointer", opacity: recalcLoading ? 0.6 : 1 }}>
                              {recalcLoading ? "RECALCULATING…" : "RECALCULATE ↺"}
                            </button>
                          </>
                        ) : (
                          <button onClick={() => { setSpecDraft({ ...spec }); setSpecEditMode(true); }} style={{ fontFamily: T.mono, fontSize: T.fontMicro, letterSpacing: "0.1em", color: "#A78BFA", background: "rgba(124,58,237,0.08)", border: `1px solid ${T.violetBorder}`, borderRadius: 4, padding: "3px 10px", cursor: "pointer" }}>
                            EDIT SPEC
                          </button>
                        )}
                      </div>
                    </div>

                    {spec.original_description && !specEditMode && (
                      <p style={{ fontSize: T.fontSmall, color: T.textMuted, fontFamily: T.sans, fontStyle: "italic", marginBottom: 12, lineHeight: 1.5, borderLeft: `2px solid ${T.border}`, paddingLeft: 10 }}>
                        "{spec.original_description}"
                      </p>
                    )}

                    <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
                      {/* Numeric fields — editable */}
                      {[
                        { key: "monthly_queries", label: "QUERIES/MO" },
                        { key: "input_tokens_per_query", label: "INPUT TOKENS" },
                        { key: "output_tokens_per_query", label: "OUTPUT TOKENS" },
                      ].map(({ key, label }) => (
                        <div key={key} style={{ background: T.bg, border: `1px solid ${T.violetBorder}`, borderRadius: 6, padding: "6px 12px", display: "flex", flexDirection: "column", gap: 4 }}>
                          <span style={{ fontFamily: T.mono, fontSize: 8, letterSpacing: "0.12em", color: "#A78BFA" }}>{label}</span>
                          {specEditMode ? (
                            <input
                              type="number"
                              value={specDraft[key] ?? ""}
                              onChange={(e) => setSpecDraft((d: any) => ({ ...d, [key]: parseInt(e.target.value) || 0 }))}
                              style={inputStyle}
                            />
                          ) : (
                            <span style={{ fontFamily: T.mono, fontSize: 14, fontWeight: 700, color: T.text }}>{fmtNum(spec[key])}</span>
                          )}
                        </div>
                      ))}

                      {/* Categorical fields — selectable */}
                      {[
                        { key: "latency_sla", label: "LATENCY", options: ["real-time", "interactive", "batch"] },
                        { key: "reasoning_complexity", label: "REASONING", options: ["low", "medium", "high"] },
                        { key: "hallucination_sensitivity", label: "HALLUCINATION RISK", options: ["low", "medium", "high"] },
                      ].map(({ key, label, options }) => (
                        <div key={key} style={{ background: T.bg, border: `1px solid ${T.border}`, borderRadius: 6, padding: "6px 12px", display: "flex", flexDirection: "column", gap: 4 }}>
                          <span style={{ fontFamily: T.mono, fontSize: 8, letterSpacing: "0.12em", color: T.textMuted }}>{label}</span>
                          {specEditMode ? (
                            <select value={specDraft[key] ?? ""} onChange={(e) => setSpecDraft((d: any) => ({ ...d, [key]: e.target.value }))} style={selectStyle}>
                              {options.map((o) => <option key={o} value={o}>{o}</option>)}
                            </select>
                          ) : (
                            <span style={{ fontFamily: T.mono, fontSize: 14, fontWeight: 700, color: "#9CA3AF" }}>{draft[key] ?? "—"}</span>
                          )}
                        </div>
                      ))}
                    </div>

                    {specEditMode && (
                      <p style={{ fontSize: 10, color: "#A78BFA", fontFamily: T.sans, marginTop: 10, lineHeight: 1.5 }}>
                        Editing bypasses the Parsing Agent — your values go directly to the cost engine.
                      </p>
                    )}
                    {!specEditMode && spec.advisory_notes && (
                      <p style={{ fontSize: 10, color: T.textMuted, fontFamily: T.sans, marginTop: 10, lineHeight: 1.5 }}>
                        <span style={{ color: "#7C3AED" }}>note: </span>{spec.advisory_notes}
                      </p>
                    )}
                  </div>
                );
              })()}

              {/* recommendation card */}
              <div style={{ borderRadius: 12, overflow: "hidden", marginBottom: 14 }}>
                {/* top bar */}
                <div style={{ background: headerBg, padding: "10px 24px", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                  <span style={{ fontFamily: T.mono, fontSize: 10, letterSpacing: "0.2em", color: headerTextColor, fontWeight: 600 }}>INFRASTRUCTURE RECOMMENDATION</span>
                  <span style={{ fontFamily: T.mono, fontSize: 10, color: headerTextColor, opacity: 0.7 }}>based on workload + cost analysis</span>
                </div>

                {/* main body */}
                <div style={{ background: T.surface, border: `1px solid ${borderColor}`, borderTop: "none", padding: "24px" }}>

                  {/* latency warning banner */}
                  {(backendRec?.latency_flag === "api_latency_risk_hard" || backendRec?.latency_flag === "api_latency_risk_soft") && (() => {
                    const isHard = backendRec.latency_flag === "api_latency_risk_hard";
                    return (
                      <div style={{ display: "flex", alignItems: "flex-start", gap: 12, background: isHard ? "#450A0A" : "#451A03", border: `1px solid ${isHard ? "#991B1B" : "#92400E"}`, borderRadius: 8, padding: "12px 16px", marginBottom: 20 }}>
                        <span style={{ fontSize: 18, flexShrink: 0 }}>{isHard ? "🚫" : "⚡"}</span>
                        <div>
                          <div style={{ fontFamily: T.mono, fontSize: T.fontMicro, letterSpacing: "0.12em", color: isHard ? "#FCA5A5" : "#FCD34D", fontWeight: 700, marginBottom: 4 }}>
                            {isHard ? "HARD LATENCY REQUIREMENT — API NOT VIABLE" : "LATENCY SLA RISK"}
                          </div>
                          <p style={{ fontSize: T.fontBody, color: isHard ? "#FECACA" : "#FDE68A", fontFamily: T.sans, lineHeight: 1.6 }}>
                            {backendRec.latency_note || "Managed API round-trip latency (300ms–3s p99) may not meet a real-time SLA. Self-hosted inference in the same datacenter typically achieves 30–200ms p99."}
                          </p>
                        </div>
                      </div>
                    );
                  })()}

                  {/* quality gap warning banner */}
                  {backendRec?.quality_gap_warning && (
                    <div style={{ display: "flex", alignItems: "flex-start", gap: 12, background: "#1E1B4B", border: "1px solid #4338CA", borderRadius: 8, padding: "12px 16px", marginBottom: 20 }}>
                      <span style={{ fontSize: 18, flexShrink: 0 }}>⚠️</span>
                      <div>
                        <div style={{ fontFamily: T.mono, fontSize: T.fontMicro, letterSpacing: "0.12em", color: "#A5B4FC", fontWeight: 700, marginBottom: 4 }}>QUALITY GAP WARNING</div>
                        <p style={{ fontSize: T.fontBody, color: "#C7D2FE", fontFamily: T.sans, lineHeight: 1.6 }}>{backendRec.quality_gap_warning}</p>
                      </div>
                    </div>
                  )}

                  {/* toolchain friction banner — shown when self-host recommendation has non-trivial porting cost */}
                  {!isApiRecommended && backendRec?.toolchain_friction && backendRec.toolchain_friction !== "none" && (
                    <div style={{
                      display: "flex", alignItems: "flex-start", gap: 12,
                      background: backendRec.toolchain_friction === "high" ? "rgba(239,68,68,0.06)" : "rgba(251,191,36,0.06)",
                      border: `1px solid ${backendRec.toolchain_friction === "high" ? "#7F1D1D" : "#78350F"}`,
                      borderRadius: 8, padding: "12px 16px", marginBottom: 20,
                    }}>
                      <span style={{ fontSize: 18, flexShrink: 0 }}>🔧</span>
                      <div>
                        <div style={{ fontFamily: T.mono, fontSize: T.fontMicro, letterSpacing: "0.12em", color: backendRec.toolchain_friction === "high" ? "#FCA5A5" : "#FCD34D", fontWeight: 700, marginBottom: 4 }}>
                          {backendRec.toolchain_friction === "high" ? "HIGH TOOLCHAIN FRICTION" : "MODERATE TOOLCHAIN FRICTION"}
                        </div>
                        <p style={{ fontSize: T.fontBody, color: backendRec.toolchain_friction === "high" ? "#FCA5A5" : "#FDE68A", fontFamily: T.sans, lineHeight: 1.6 }}>
                          {backendRec.toolchain_friction_note || (backendRec.toolchain_friction === "high"
                            ? "This combination requires non-trivial toolchain porting (4–12 weeks). Consider a CUDA provider at a modest cost premium."
                            : "This combination requires porting to a custom SDK (1–4 weeks of engineering). Official or community support exists for this model.")}
                        </p>
                      </div>
                    </div>
                  )}

                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 0, marginBottom: 24 }}>

                    {/* verdict */}
                    <div style={{ borderRight: `1px solid ${T.border}`, paddingRight: 24 }}>
                      <div style={{ fontFamily: T.mono, fontSize: T.fontMicro, letterSpacing: "0.16em", color: T.textMuted, marginBottom: 10 }}>VERDICT</div>
                      <div style={{ fontFamily: T.sans, fontSize: 36, fontWeight: 800, letterSpacing: "-0.03em", color: verdictColor, lineHeight: 1, marginBottom: 8 }}>
                        {isApiRecommended ? "API" : isOpenWeightRecommended ? "Open-Weight" : "Self-hosted"}
                      </div>
                    </div>

                    {/* rationale */}
                    <div style={{ borderRight: `1px solid ${T.border}`, padding: "0 24px" }}>
                      <div style={{ fontFamily: T.mono, fontSize: T.fontMicro, letterSpacing: "0.16em", color: T.textMuted, marginBottom: 10 }}>WHY</div>
                      <p style={{ fontSize: T.fontBody, lineHeight: 1.65, color: "#D1D5DB", fontFamily: T.sans }}>
                        {backendRec?.rationale ?? "Compare the tables below to see API and GPU provider costs side-by-side across growth scenarios."}
                      </p>
                    </div>

                    {/* migration trigger / breakeven */}
                    <div style={{ paddingLeft: 24 }}>
                      <div style={{ fontFamily: T.mono, fontSize: T.fontMicro, letterSpacing: "0.16em", color: T.amber, marginBottom: 10 }}>BREAKEVEN</div>
                      <p style={{ fontSize: T.fontBody, lineHeight: 1.65, color: "#FCD34D", fontFamily: T.sans }}>
                        {backendRec?.crossover_note ?? "See crossover analysis in the tables below."}
                      </p>
                      {backendRec?.confidence_score != null && (
                        <div style={{ marginTop: 12, display: "flex", alignItems: "center", gap: 8 }}>
                          <div style={{ flex: 1, height: 3, background: T.border, borderRadius: 2 }}>
                            <div style={{ width: `${(backendRec.confidence_score * 100).toFixed(0)}%`, height: "100%", background: `linear-gradient(90deg, ${T.violet}, ${T.amber})`, borderRadius: 2 }} />
                          </div>
                          <span style={{ fontFamily: T.mono, fontSize: T.fontMicro, color: T.textMuted, whiteSpace: "nowrap" }}>
                            {(backendRec.confidence_score * 100).toFixed(0)}% confidence
                          </span>
                        </div>
                      )}
                    </div>
                  </div>

                  {/* key cost numbers */}
                  {result.scenarios.length > 0 && (
                    <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 10, borderTop: `1px solid ${T.border}`, paddingTop: 20 }}>
                      {[
                        { label: "Cheapest API now", value: fmtCost(result.scenarios[0].cheapest_api_model.monthly_cost) + "/mo", sub: result.scenarios[0].cheapest_api_model.display_name || result.scenarios[0].cheapest_api_model.model_key, color: T.violet },
                        { label: "Open-weight GPU now", value: fmtCost(result.scenarios[0].cheapest_open_weight_option?.monthly_cost ?? result.scenarios[0].cheapest_gpu_provider?.monthly_cost) + "/mo", sub: result.scenarios[0].cheapest_open_weight_option?.display_name ?? result.scenarios[0].cheapest_gpu_provider?.display_name ?? "—", color: "#34D399" },
                        { label: "API at 5x", value: fmtCost(result.scenarios[2].cheapest_api_model.monthly_cost) + "/mo", sub: "projected growth", color: T.violet },
                        { label: "Open-weight GPU at 5x", value: fmtCost(result.scenarios[2].cheapest_open_weight_option?.monthly_cost ?? result.scenarios[2].cheapest_gpu_provider?.monthly_cost) + "/mo", sub: result.scenarios[2].cheapest_open_weight_option?.quality_tier ? `${result.scenarios[2].cheapest_open_weight_option.quality_tier} quality tier` : "projected growth", color: "#34D399" },
                      ].map((item) => (
                        <div key={item.label} style={{ background: T.bg, borderRadius: 8, padding: "12px 14px", border: `1px solid ${T.border}` }}>
                          <div style={{ fontFamily: T.mono, fontSize: T.fontMicro, color: T.textMuted, letterSpacing: "0.1em", marginBottom: 6 }}>{item.label.toUpperCase()}</div>
                          <div style={{ fontFamily: T.mono, fontSize: 22, fontWeight: 700, color: item.color, marginBottom: 4 }}>{item.value}</div>
                          <div style={{ fontSize: T.fontSmall, color: T.textMuted, fontFamily: T.sans }}>{item.sub}</div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>

              {/* ── DEPLOYMENT DECISION PATH — two-step hierarchy ── */}
              {result.scenarios.length > 0 && (() => {
                const s0 = result.scenarios[0];
                const cheapestApi = s0.cheapest_api_model;
                const cheapestOW = s0.cheapest_open_weight_option;
                const cheapestGpu = s0.cheapest_gpu_provider;
                // The cheapest self-host option is always the open-weight option when available
                const cheapestSelfHost = cheapestOW ?? cheapestGpu;

                const optionBoxBase: React.CSSProperties = {
                  borderRadius: 8, padding: "14px 16px", flex: 1,
                };
                const proStyle: React.CSSProperties = { display: "flex", gap: 6, marginBottom: 3 };
                const proText = (s: string) => (
                  <div style={proStyle}>
                    <span style={{ color: "#34D399", fontSize: T.fontSmall, lineHeight: 1 }}>+</span>
                    <span style={{ fontSize: T.fontSmall, color: "#6EE7B7", fontFamily: T.sans }}>{s}</span>
                  </div>
                );
                const conText = (s: string) => (
                  <div style={proStyle}>
                    <span style={{ color: "#F87171", fontSize: T.fontSmall, lineHeight: 1 }}>−</span>
                    <span style={{ fontSize: T.fontSmall, color: "#FCA5A5", fontFamily: T.sans }}>{s}</span>
                  </div>
                );

                return (
                  <div style={{ marginBottom: 14 }}>
                    {/* ── Step 1: API vs Self-Host ── */}
                    <Card style={{ marginBottom: 6 }}>
                      <div style={{ fontFamily: T.mono, fontSize: T.fontMicro, letterSpacing: "0.18em", color: T.textMuted, marginBottom: 14 }}>
                        STEP 1 · PRIMARY DECISION — MANAGED API vs SELF-HOSTING
                      </div>
                      <div style={{ display: "flex", gap: 12, alignItems: "stretch" }}>

                        {/* Option A: Managed API */}
                        <div style={{
                          ...optionBoxBase,
                          background: isApiRecommended ? T.violetDim : T.bg,
                          border: `1px solid ${isApiRecommended ? T.violetBorder : T.border}`,
                        }}>
                          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 8 }}>
                            <span style={{ fontFamily: T.mono, fontSize: T.fontMicro, letterSpacing: "0.12em", color: isApiRecommended ? "#A78BFA" : T.textMuted }}>
                              OPTION A · MANAGED API
                            </span>
                            {isApiRecommended && (
                              <span style={{ fontFamily: T.mono, fontSize: T.fontMicro, color: "#A78BFA", background: T.violetDim, border: `1px solid ${T.violetBorder}`, borderRadius: 3, padding: "1px 6px" }}>
                                RECOMMENDED
                              </span>
                            )}
                          </div>
                          <div style={{ fontFamily: T.mono, fontSize: 28, fontWeight: 700, color: isApiRecommended ? "#A78BFA" : T.text, marginBottom: 2 }}>
                            {fmtCost(cheapestApi.monthly_cost)}<span style={{ fontSize: T.fontSmall, fontWeight: 400, color: T.textMuted }}>/mo</span>
                          </div>
                          <div style={{ fontSize: T.fontSmall, color: T.textMuted, fontFamily: T.sans, marginBottom: 10 }}>
                            {cheapestApi.display_name ?? cheapestApi.model_key}
                          </div>
                          {proText("Zero ops overhead")}
                          {proText("Instant elasticity")}
                          {conText("Per-token cost scales linearly with volume")}
                        </div>

                        {/* VS divider */}
                        <div style={{ display: "flex", alignItems: "center", padding: "0 4px" }}>
                          <span style={{ fontFamily: T.mono, fontSize: T.fontSmall, color: T.textDim, letterSpacing: "0.1em" }}>VS</span>
                        </div>

                        {/* Option B: Self-Host */}
                        <div style={{
                          ...optionBoxBase,
                          background: !isApiRecommended ? "rgba(52,211,153,0.06)" : T.bg,
                          border: `1px solid ${!isApiRecommended ? "#065F46" : T.border}`,
                        }}>
                          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 8 }}>
                            <span style={{ fontFamily: T.mono, fontSize: T.fontMicro, letterSpacing: "0.12em", color: !isApiRecommended ? "#34D399" : T.textMuted }}>
                              OPTION B · SELF-HOST ON GPU
                            </span>
                            {!isApiRecommended && (
                              <span style={{ fontFamily: T.mono, fontSize: T.fontMicro, color: "#34D399", background: "rgba(52,211,153,0.1)", border: "1px solid #065F46", borderRadius: 3, padding: "1px 6px" }}>
                                RECOMMENDED
                              </span>
                            )}
                          </div>
                          <div style={{ fontFamily: T.mono, fontSize: 28, fontWeight: 700, color: !isApiRecommended ? "#34D399" : T.text, marginBottom: 2 }}>
                            {cheapestSelfHost ? fmtCost(cheapestSelfHost.monthly_cost) : "—"}
                            <span style={{ fontSize: T.fontSmall, fontWeight: 400, color: T.textMuted }}>/mo</span>
                          </div>
                          <div style={{ fontSize: T.fontSmall, color: T.textMuted, fontFamily: T.sans, marginBottom: 10 }}>
                            {cheapestSelfHost?.display_name ?? "Best self-host option"}
                          </div>
                          {proText("No per-token cost at scale")}
                          {proText("Full data control and low latency")}
                          {conText("Ops complexity + ML infra team required")}
                        </div>
                      </div>
                    </Card>

                    {/* ── Step 2: Which self-host approach? (indented under Option B) ── */}
                    <div style={{ marginLeft: 20, borderLeft: `2px solid rgba(52,211,153,0.18)`, paddingLeft: 14 }}>
                      <Card>
                        <div style={{ fontFamily: T.mono, fontSize: T.fontMicro, letterSpacing: "0.18em", color: "#34D399", marginBottom: 4 }}>
                          STEP 2 · IF SELF-HOSTING — WHICH MODEL APPROACH?
                        </div>
                        <p style={{ fontSize: T.fontSmall, color: T.textMuted, fontFamily: T.sans, marginBottom: 14, lineHeight: 1.6 }}>
                          Open-weight models (Llama, Qwen) eliminate per-token cost entirely — you only pay for GPU hours.
                          Running a closed/proprietary model on rented GPU is rarely cheaper because you still pay per-token or per-seat fees on top of hardware.
                        </p>
                        <div style={{ display: "flex", gap: 12, alignItems: "stretch" }}>

                          {/* B1: Open-weight */}
                          <div style={{
                            ...optionBoxBase,
                            background: "rgba(52,211,153,0.06)",
                            border: "1px solid #065F46",
                          }}>
                            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 8 }}>
                              <span style={{ fontFamily: T.mono, fontSize: T.fontMicro, letterSpacing: "0.12em", color: "#34D399" }}>B1 · OPEN-WEIGHT ON GPU</span>
                              <span style={{ fontFamily: T.mono, fontSize: T.fontMicro, color: "#34D399", background: "rgba(52,211,153,0.1)", border: "1px solid #065F46", borderRadius: 3, padding: "1px 6px" }}>PREFERRED</span>
                            </div>
                            {cheapestOW ? (
                              <>
                                <div style={{ fontFamily: T.mono, fontSize: 24, fontWeight: 700, color: "#34D399", marginBottom: 2 }}>
                                  {fmtCost(cheapestOW.monthly_cost)}<span style={{ fontSize: T.fontSmall, fontWeight: 400, color: T.textMuted }}>/mo</span>
                                </div>
                                <div style={{ fontSize: T.fontSmall, color: T.textMuted, fontFamily: T.sans, marginBottom: 6 }}>{cheapestOW.display_name}</div>
                                <div style={{ fontSize: T.fontMicro, color: "#6EE7B7", fontFamily: T.mono, marginBottom: 6 }}>
                                  {cheapestOW.quality_tier} quality · {cheapestOW.estimated_gpu_count} GPU{cheapestOW.estimated_gpu_count !== 1 ? "s" : ""}
                                </div>
                                {cheapestOW.toolchain_friction && cheapestOW.toolchain_friction !== "none" && (() => {
                                  const fc = cheapestOW.toolchain_friction === "high" ? { bg: "rgba(239,68,68,0.08)", border: "#7F1D1D", text: "#FCA5A5", label: "HIGH FRICTION" } : { bg: "rgba(251,191,36,0.08)", border: "#78350F", text: "#FCD34D", label: "MODERATE FRICTION" };
                                  return (
                                    <div style={{ background: fc.bg, border: `1px solid ${fc.border}`, borderRadius: 5, padding: "6px 8px", marginBottom: 6 }}>
                                      <div style={{ fontFamily: T.mono, fontSize: T.fontMicro, color: fc.text, letterSpacing: "0.1em", marginBottom: 2 }}>{fc.label}</div>
                                      <div style={{ fontSize: T.fontMicro, color: fc.text, fontFamily: T.sans, lineHeight: 1.5 }}>{cheapestOW.toolchain_friction_note}</div>
                                    </div>
                                  );
                                })()}
                              </>
                            ) : (
                              <div style={{ fontSize: T.fontSmall, color: T.textMuted, fontFamily: T.sans }}>No open-weight data</div>
                            )}
                            {proText("Zero per-token cost")}
                            {proText("Open license (commercial OK)")}
                            {conText("Model quality varies by size")}
                          </div>

                          {/* vs */}
                          <div style={{ display: "flex", alignItems: "center", padding: "0 4px" }}>
                            <span style={{ fontFamily: T.mono, fontSize: T.fontMicro, color: T.textDim, letterSpacing: "0.1em" }}>vs</span>
                          </div>

                          {/* B2: GPU infra only (closed model) */}
                          <div style={{ ...optionBoxBase, background: T.bg, border: `1px solid ${T.border}` }}>
                            <div style={{ marginBottom: 8 }}>
                              <span style={{ fontFamily: T.mono, fontSize: T.fontMicro, letterSpacing: "0.12em", color: T.textMuted }}>
                                B2 · GPU INFRA (CLOSED MODEL)
                              </span>
                            </div>
                            <div style={{ fontFamily: T.mono, fontSize: 24, fontWeight: 700, color: T.text, marginBottom: 2 }}>
                              {fmtCost(cheapestGpu.monthly_cost)}<span style={{ fontSize: T.fontSmall, fontWeight: 400, color: T.textMuted }}>/mo</span>
                            </div>
                            <div style={{ fontSize: T.fontSmall, color: T.textMuted, fontFamily: T.sans, marginBottom: 8 }}>
                              {cheapestGpu.display_name} · hardware only
                            </div>
                            {conText("Still requires model license or per-token fees on top")}
                            {conText("Rarely cheaper than Option A at these scales")}
                            {proText("Supports any proprietary model via API gateway")}
                          </div>
                        </div>
                      </Card>
                    </div>
                  </div>
                );
              })()}

              {/* ── DETAILED BREAKDOWN ── */}
              <div style={{ fontFamily: T.mono, fontSize: T.fontMicro, letterSpacing: "0.18em", color: T.textDim, marginBottom: 10, marginTop: 8 }}>
                DETAILED BREAKDOWN
              </div>

              {/* cost projection chart */}
              <Card style={{ marginBottom: 14 }}>
                <div style={{ fontFamily: T.mono, fontSize: T.fontMicro, letterSpacing: "0.18em", color: T.textMuted, marginBottom: 4 }}>OPTION A · MANAGED API — ALL MODELS</div>
                <p style={{ fontSize: T.fontSmall, color: T.textMuted, fontFamily: T.sans, marginBottom: 14, lineHeight: 1.6 }}>Monthly cost across all API models at current, 2x, and 5x query volume. Cost-per-query is the key comparison metric.</p>
                <div style={{ overflowX: "auto" }}>
                  <table style={{ width: "100%", borderCollapse: "collapse", fontSize: T.fontSmall, fontFamily: T.mono }}>
                    <thead>
                      <tr style={{ borderBottom: `1px solid ${T.border}` }}>
                        {["Model", "Provider", "Tier", "Current / mo", "¢ per query", "2x / mo", "5x / mo"].map((h) => (
                          <th key={h} style={{ padding: "10px 12px", textAlign: "left", fontSize: T.fontSmall, letterSpacing: "0.06em", color: "#9CA3AF", fontWeight: 600 }}>{h}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {result.scenarios[0].all_api_models.map((model: any) => {
                        const s2 = result.scenarios[1]?.all_api_models?.find((m: any) => m.model_key === model.model_key);
                        const s5 = result.scenarios[2]?.all_api_models?.find((m: any) => m.model_key === model.model_key);
                        const tierColor: Record<string, string> = { budget: "#8B5CF6", premium: "#3B82F6", frontier: "#EC4899", unknown: T.textMuted };
                        const tc = tierColor[model.tier] ?? T.textMuted;
                        return (
                          <tr key={model.model_key} style={{ borderBottom: `1px solid ${T.border}` }}>
                            <td style={{ padding: "11px 12px", color: T.text, fontWeight: 600 }}>{model.display_name || model.model_key.replace(/_/g, " ")}</td>
                            <td style={{ padding: "11px 12px", color: T.textMuted, fontSize: T.fontSmall }}>{model.provider}</td>
                            <td style={{ padding: "11px 12px" }}>
                              <span style={{ fontSize: T.fontMicro, padding: "2px 7px", borderRadius: 3, background: `${tc}18`, border: `1px solid ${tc}40`, color: tc, letterSpacing: "0.08em" }}>{model.tier}</span>
                            </td>
                            <td style={{ padding: "11px 12px", color: T.text, fontWeight: 600 }}>{fmtCost(model.monthly_cost)}</td>
                            <td style={{ padding: "11px 12px", color: T.amber, fontWeight: 600 }}>{(model.cost_per_query * 100).toFixed(4)}¢</td>
                            <td style={{ padding: "11px 12px", color: T.textMuted }}>{s2 ? fmtCost(s2.monthly_cost) : "—"}</td>
                            <td style={{ padding: "11px 12px", color: T.textMuted }}>{s5 ? fmtCost(s5.monthly_cost) : "—"}</td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </Card>

              {/* GPU provider costs table */}
              <Card style={{ marginBottom: 14 }}>
                <div style={{ fontFamily: T.mono, fontSize: T.fontMicro, letterSpacing: "0.18em", color: T.textMuted, marginBottom: 4 }}>SELF-HOSTED GPU INFRASTRUCTURE</div>
                <p style={{ fontSize: T.fontSmall, color: T.textMuted, fontFamily: T.sans, marginBottom: 14, lineHeight: 1.6 }}>Monthly cost to self-host across CoreWeave, Lambda Labs, AWS Trainium2, and GCP TPU v5e at current query volume.</p>
                <div style={{ overflowX: "auto" }}>
                  <table style={{ width: "100%", borderCollapse: "collapse", fontSize: T.fontSmall, fontFamily: T.mono }}>
                    <thead>
                      <tr style={{ borderBottom: `1px solid ${T.border}` }}>
                        {["Provider", "Chip", "GPUs needed", "Monthly cost", "¢ per query", "2x cost", "5x cost"].map((h) => (
                          <th key={h} style={{ padding: "10px 12px", textAlign: "left", fontSize: T.fontSmall, letterSpacing: "0.06em", color: "#9CA3AF", fontWeight: 600 }}>{h}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {result.scenarios[0].gpu_providers.map((gpu: any) => {
                        const g2 = result.scenarios[1]?.gpu_providers?.find((g: any) => g.provider_key === gpu.provider_key);
                        const g5 = result.scenarios[2]?.gpu_providers?.find((g: any) => g.provider_key === gpu.provider_key);
                        return (
                          <tr key={gpu.provider_key} style={{ borderBottom: `1px solid ${T.border}` }}>
                            <td style={{ padding: "11px 12px" }}>
                              <div style={{ color: T.text, fontWeight: 600 }}>{gpu.display_name}</div>
                              <div style={{ fontSize: T.fontMicro, color: T.textMuted, marginTop: 2 }}>{gpu.provider}</div>
                            </td>
                            <td style={{ padding: "11px 12px", color: T.textMuted, fontSize: T.fontSmall }}>{gpu.chip}</td>
                            <td style={{ padding: "11px 12px", color: T.text }}>{gpu.estimated_gpu_count}</td>
                            <td style={{ padding: "11px 12px", color: T.amber, fontWeight: 600 }}>{fmtCost(gpu.monthly_cost)}</td>
                            <td style={{ padding: "11px 12px", color: T.amber }}>{(gpu.cost_per_query * 100).toFixed(4)}¢</td>
                            <td style={{ padding: "11px 12px", color: T.textMuted }}>{g2 ? fmtCost(g2.monthly_cost) : "—"}</td>
                            <td style={{ padding: "11px 12px", color: T.textMuted }}>{g5 ? fmtCost(g5.monthly_cost) : "—"}</td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </Card>

              {/* open-weight options table */}
              {result.scenarios[0].open_weight_options?.length > 0 && (
                <Card style={{ marginBottom: 14 }}>
                  <div style={{ fontFamily: T.mono, fontSize: T.fontMicro, letterSpacing: "0.18em", color: "#34D399", marginBottom: 4 }}>OPEN-WEIGHT MODELS ON GPU</div>
                  <p style={{ fontSize: T.fontSmall, color: T.textMuted, fontFamily: T.sans, marginBottom: 14, lineHeight: 1.6 }}>
                    Self-hosting open-weight models (Llama, Qwen) — no per-token cost, only GPU hours. Compare to managed API for true cost crossover.
                  </p>
                  <div style={{ overflowX: "auto" }}>
                    <table style={{ width: "100%", borderCollapse: "collapse", fontSize: T.fontSmall, fontFamily: T.mono }}>
                      <thead>
                        <tr style={{ borderBottom: `1px solid ${T.border}` }}>
                          {["Model", "GPU Provider", "Quality", "Toolchain", "GPUs", "Monthly cost", "¢ per query", "2x cost", "5x cost"].map((h) => (
                            <th key={h} style={{ padding: "10px 12px", textAlign: "left", fontSize: T.fontSmall, letterSpacing: "0.06em", color: "#9CA3AF", fontWeight: 600 }}>{h}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {result.scenarios[0].open_weight_options.map((opt: any) => {
                          const o2 = result.scenarios[1]?.open_weight_options?.find((o: any) => o.option_key === opt.option_key);
                          const o5 = result.scenarios[2]?.open_weight_options?.find((o: any) => o.option_key === opt.option_key);
                          const tierColor: Record<string, string> = { budget: "#34D399", mid: "#60A5FA", frontier: "#EC4899", unknown: T.textMuted };
                          const tc = tierColor[opt.quality_tier] ?? T.textMuted;
                          return (
                            <tr key={opt.option_key} style={{ borderBottom: `1px solid ${T.border}` }}>
                              <td style={{ padding: "11px 12px", color: T.text, fontWeight: 600 }}>{opt.model_display_name}</td>
                              <td style={{ padding: "11px 12px", color: T.textMuted, fontSize: T.fontSmall }}>{opt.provider_display_name}</td>
                              <td style={{ padding: "11px 12px" }}>
                                <span style={{ fontSize: T.fontMicro, padding: "2px 7px", borderRadius: 3, background: `${tc}18`, border: `1px solid ${tc}40`, color: tc, letterSpacing: "0.08em" }}>{opt.quality_tier}</span>
                              </td>
                              <td style={{ padding: "11px 12px" }}>{(() => {
                                const f = opt.toolchain_friction ?? "none";
                                const fStyle: Record<string, { bg: string; border: string; color: string }> = {
                                  none:     { bg: "rgba(52,211,153,0.08)",  border: "#065F46", color: "#34D399" },
                                  moderate: { bg: "rgba(251,191,36,0.08)",  border: "#78350F", color: "#FCD34D" },
                                  high:     { bg: "rgba(239,68,68,0.08)",   border: "#7F1D1D", color: "#FCA5A5" },
                                };
                                const fs = fStyle[f] ?? fStyle.none;
                                return (
                                  <span title={opt.toolchain_friction_note ?? ""} style={{ fontSize: T.fontMicro, padding: "2px 7px", borderRadius: 3, background: fs.bg, border: `1px solid ${fs.border}`, color: fs.color, letterSpacing: "0.08em", cursor: f !== "none" ? "help" : "default" }}>
                                    {f}
                                  </span>
                                );
                              })()}</td>
                              <td style={{ padding: "11px 12px", color: T.text }}>{opt.estimated_gpu_count}</td>
                              <td style={{ padding: "11px 12px", color: "#34D399", fontWeight: 600 }}>{fmtCost(opt.monthly_cost)}</td>
                              <td style={{ padding: "11px 12px", color: "#34D399" }}>{(opt.cost_per_query * 100).toFixed(4)}¢</td>
                              <td style={{ padding: "11px 12px", color: T.textMuted }}>{o2 ? fmtCost(o2.monthly_cost) : "—"}</td>
                              <td style={{ padding: "11px 12px", color: T.textMuted }}>{o5 ? fmtCost(o5.monthly_cost) : "—"}</td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                </Card>
              )}

              {/* cost projection chart */}
              <Card>
                <div style={{ fontFamily: T.mono, fontSize: T.fontMicro, letterSpacing: "0.18em", color: T.textMuted, marginBottom: 6 }}>COST PROJECTION</div>
                <div style={{ fontSize: T.fontHeading, fontWeight: 700, color: T.text, fontFamily: T.sans, marginBottom: 6 }}>API tiers vs GPU providers across growth scenarios</div>
                <p style={{ fontSize: T.fontSmall, color: T.textMuted, fontFamily: T.sans, lineHeight: 1.6, marginBottom: 20 }}>Monthly cost comparison across all API tiers and all GPU providers as workload scale grows.</p>
                <div style={{ display: "flex", gap: 16, flexWrap: "wrap", marginBottom: 16 }}>
                  {costChartSeries.map((series) => (
                    <div key={series.dataKey} style={{ display: "flex", alignItems: "center", gap: 6 }}>
                      <div style={{ width: 20, height: 2, background: series.color, borderRadius: 1 }} />
                      <span style={{ fontFamily: T.mono, fontSize: T.fontMicro, color: T.textMuted }}>{series.label}</span>
                    </div>
                  ))}
                </div>
                <div style={{ height: 300, background: T.bg, border: `1px solid ${T.border}`, borderRadius: 8, padding: "16px 8px 8px" }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={costChartData} margin={{ top: 4, right: 20, left: 8, bottom: 4 }}>
                      <CartesianGrid stroke="rgba(255,255,255,0.04)" strokeDasharray="4 4" vertical={false} />
                      <XAxis dataKey="scenario" tick={{ fill: "#9CA3AF", fontSize: 12, fontFamily: "DM Mono, monospace" }} axisLine={{ stroke: "rgba(255,255,255,0.06)" }} tickLine={false} />
                      <YAxis tick={{ fill: "#9CA3AF", fontSize: 12, fontFamily: "DM Mono, monospace" }} axisLine={false} tickLine={false} tickFormatter={(v) => fmtCost(Number(v))} />
                      <Tooltip
                        contentStyle={{ backgroundColor: T.surface, border: "1px solid rgba(255,255,255,0.08)", borderRadius: 8, fontSize: T.fontSmall, fontFamily: "DM Mono, monospace" }}
                        labelStyle={{ color: T.violet, marginBottom: 6 }}
                        itemStyle={{ color: "#D1D5DB" }}
                        formatter={(value, name) => [fmtCost(Number(value)), costChartLabelByKey[String(name)] ?? String(name)]}
                      />
                      {costChartSeries.map((series) => (
                        <Line key={series.dataKey} type="monotone" dataKey={series.dataKey} name={series.dataKey}
                          stroke={series.color} strokeWidth={2}
                          dot={{ fill: series.color, strokeWidth: 0, r: 3 }}
                          activeDot={{ r: 5, fill: series.color, stroke: T.bg, strokeWidth: 2 }}
                          connectNulls={false}
                        />
                      ))}
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </Card>
            </>
          )}
        </div>

        {/* ── FOOTER ── */}
        <footer style={{ borderTop: `1px solid ${T.border}`, padding: "20px 40px", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
          <div style={{ fontFamily: T.mono, fontSize: T.fontMicro, letterSpacing: "0.14em", color: T.textDim }}>AI INFRA COST SIMULATOR</div>
          <div style={{ fontFamily: T.mono, fontSize: T.fontMicro, letterSpacing: "0.14em", color: T.textDim }}>Google × Kaggle Agentic AI Bootcamp</div>
        </footer>
      </main>
    </>
  );
}