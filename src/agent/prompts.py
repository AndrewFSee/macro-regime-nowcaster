"""LLM prompt templates for the macro narrative agent.

Contains the system-level instruction prompt and a Jinja2-style
narrative template that is filled with nowcast results before being
sent to the language model.
"""

MACRO_ANALYST_SYSTEM_PROMPT: str = """You are a senior macroeconomic analyst at a global asset management firm. \
Your role is to synthesise quantitative economic signals with Federal Reserve communications to produce \
concise, actionable regime narratives for portfolio managers.

You will be given:
1. A quantitative nowcast result including the current economic regime, regime probabilities, \
   latent factor values, and a GDP growth estimate.
2. Excerpts from recent Federal Reserve communications (FOMC minutes, statements, Beige Book, speeches).

Your output must include:
- **Summary** (2-3 sentences): State the current regime and what the key macro data signals are showing.
- **Key Drivers** (3-5 bullet points): The main economic forces driving the regime classification.
- **Fed Alignment** (1-2 sentences): Whether Fed language corroborates or contradicts the quant signal.
- **Risk Flags** (2-3 bullet points): The most important tail risks or potential regime transitions.

Style guidelines:
- Be concise and precise. No filler language.
- Use specific numbers from the input data.
- Quantify uncertainty where possible.
- Flag if data quality or recency is a concern.
"""

NARRATIVE_TEMPLATE: str = """## Nowcast Input Data

**Current Regime:** {current_regime}
**Regime Probabilities:**
{regime_probs_formatted}

**GDP Nowcast:** {gdp_nowcast:.2f}% annualised (90% CI: [{gdp_ci_lower:.2f}%, {gdp_ci_upper:.2f}%])

**Latest Factor Values:**
{factor_values_formatted}

**Nowcast Timestamp:** {timestamp}

---

## Federal Reserve Communications (Recent)

{fed_text}

---

Based on the above, provide your macro regime narrative analysis.
"""


def format_narrative_prompt(nowcast_result: dict, fed_documents: list[dict]) -> str:
    """Render the narrative prompt template with concrete values.

    Args:
        nowcast_result: Dict from ``NowcastResult.to_dict()``.
        fed_documents: List of dicts with keys ``date``, ``document_type``,
            ``title``, ``text``.

    Returns:
        Formatted prompt string ready to send to the LLM.
    """
    regime_probs = nowcast_result.get("regime_probabilities", {})
    regime_probs_str = "\n".join(
        f"  - {label}: {prob:.1%}" for label, prob in regime_probs.items()
    )

    factor_values = nowcast_result.get("factor_values", {})
    factor_values_str = "\n".join(
        f"  - {name}: {val:.3f}" for name, val in factor_values.items()
    )

    # Truncate Fed text to keep prompt within token budget
    fed_excerpts = []
    for doc in fed_documents[:3]:  # at most 3 documents
        excerpt = doc.get("text", "")[:1500]
        fed_excerpts.append(
            f"[{doc.get('document_type', 'unknown').upper()}] "
            f"{doc.get('date', '')} — {doc.get('title', '')}\n{excerpt}"
        )
    fed_text = "\n\n---\n\n".join(fed_excerpts) if fed_excerpts else "No recent Fed documents available."

    return NARRATIVE_TEMPLATE.format(
        current_regime=nowcast_result.get("current_regime", "unknown"),
        regime_probs_formatted=regime_probs_str,
        gdp_nowcast=nowcast_result.get("gdp_nowcast", 0.0),
        gdp_ci_lower=nowcast_result.get("gdp_ci_lower", 0.0),
        gdp_ci_upper=nowcast_result.get("gdp_ci_upper", 0.0),
        factor_values_formatted=factor_values_str,
        timestamp=nowcast_result.get("timestamp", ""),
        fed_text=fed_text,
    )
