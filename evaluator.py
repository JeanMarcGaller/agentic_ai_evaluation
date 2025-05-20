from dotenv import load_dotenv
load_dotenv()  # Load environment variables

from langchain_openai import ChatOpenAI
from langchain.evaluation import load_evaluator
from langchain.prompts import PromptTemplate

# --- Initialize evaluation LLM ---
llm = ChatOpenAI(model="gpt-4")  # Use GPT-4 for consistent evaluation

# --- Load single-response evaluators ---
helpfulness_eval = load_evaluator("criteria", llm=llm, config={"criteria": "helpfulness"})
correctness_eval = load_evaluator("criteria", llm=llm, config={"criteria": "correctness"})
relevance_eval = load_evaluator("criteria", llm=llm, config={"criteria": "relevance"})
conciseness_eval = load_evaluator("criteria", llm=llm, config={"criteria": "conciseness"})
coherence_eval = load_evaluator("criteria", llm=llm, config={"criteria": "coherence"})

# --- Pairwise evaluator with strict [[A]]/[[B]]/[[C]] format ---
pairwise_prompt = PromptTemplate.from_template("""
You are a strict evaluator comparing two assistant responses.

Evaluation criterion: {criteria}

Question:
{input}

Answer A:
{prediction}

Answer B:
{prediction_b}

Evaluate which answer is better overall, based on the criterion above.

Respond ONLY with:
[[A]] → if Answer A is better
[[B]] → if Answer B is better
[[C]] → if both are equally good

No explanation. Only respond with [[A]], [[B]], or [[C]].
""")

pairwise_eval = load_evaluator(
    "pairwise_string",
    llm=llm,
    config={"criteria": "overall"},
    prompt=pairwise_prompt
)

# --- Evaluation wrapper function ---
def evaluate_pairwise(question, responder, revisor):
    """
    Evaluates two responses (responder and revisor) to the same question.

    Returns:
        dict: A dictionary of evaluation scores and the pairwise decision.
    """
    evaluations = {
        "helpfulness_responder": helpfulness_eval.evaluate_strings(
            input=question, prediction=responder),
        "helpfulness_revisor": helpfulness_eval.evaluate_strings(
            input=question, prediction=revisor),
        "correctness_responder": correctness_eval.evaluate_strings(
            input=question, prediction=responder),
        "correctness_revisor": correctness_eval.evaluate_strings(
            input=question, prediction=revisor),
        "relevance_responder": relevance_eval.evaluate_strings(
            input=question, prediction=responder),
        "relevance_revisor": relevance_eval.evaluate_strings(
            input=question, prediction=revisor),
        "conciseness_responder": conciseness_eval.evaluate_strings(
            input=question, prediction=responder),
        "conciseness_revisor": conciseness_eval.evaluate_strings(
            input=question, prediction=revisor),
        "coherence_responder": coherence_eval.evaluate_strings(
            input=question, prediction=responder),
        "coherence_revisor": coherence_eval.evaluate_strings(
            input=question, prediction=revisor),
    }

    # --- Pairwise string evaluation with error handling ---
    try:
        pairwise_result = pairwise_eval.evaluate_string_pairs(
            input=question,
            prediction=responder,
            prediction_b=revisor
        )
        evaluations["pairwise_winner"] = pairwise_result.get("value")
        evaluations["pairwise_reasoning"] = pairwise_result.get("reasoning", "")
    except ValueError as e:
        evaluations["pairwise_winner"] = "Invalid"
        evaluations["pairwise_reasoning"] = f"⚠️ Format error: {str(e)}"

    return evaluations
