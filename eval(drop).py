#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import math
import argparse
from functools import partial
from collections import Counter
from typing import Any, Dict, List, Tuple, Callable
import torch
from evaluate import load
from rich.table import Table
from rich.console import Console
from rich.progress import Progress
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification

console = Console()


def is_agree(prediction: bool, reference: bool) -> bool:
    """Check whether the predicted and reference sentences are the same."""
    return True if prediction == reference else False

def geometric_mean(score1, score2):
    # compute the geometric mean of the two scores
    matched_score = (score1 * score2) ** (1 / 2)
    return matched_score


def cal_f1(correct_num, predict_num, reference_num):
    # compute sample-level precision
    # num of correct predictions / num of predictions
    sample_precision = (
        correct_num / predict_num
        if predict_num > 0
        else 0.0
    )
    # compute sample-level recall
    # num of correct predictions / num of references
    sample_recall = (
        correct_num / reference_num
        if reference_num > 0
        else 0.0
    )
    sample_recall = 1.0 if sample_recall > 1.0 else sample_recall
    # compute sample-level f1
    sample_f1 = (
        (2 * sample_precision * sample_recall)
        / (sample_precision + sample_recall)
        if sample_precision + sample_recall > 0
        else 0.0
    )
    return sample_f1

def compute_meteor(predictions: List[str], references: List[str]) -> List[float]:
    """Compute the METEOR score between the predicted and
    reference sentences.
    """
    return [
        meteor_score(
            hypothesis=word_tokenize(pred), references=[word_tokenize(ref)]
        )
        for pred, ref in zip(predictions, references)
    ]

def compute_semantic_equivalence(predictions: List[str], references: List[str]) -> List[float]:
    """Compute the semantic equivalence between the predicted and reference sentences."""
    with torch.no_grad():
        inputs = tokenizer(
            predictions,
            references,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        return probs[:, 1].tolist()



def eval_question_parsing(predictions: List[str], references: List[str], threshold):
    """Compute exact match numbers between the predicted and reference. Then calculate the F1 score.
    """
    predictions = [pred.lower() for pred in predictions]
    references = [ref.lower() for ref in references]
    correct_num = 0.0
    pred_num = len(predictions)
    ref_num = len(references)
    for pred in predictions:
        token_score = compute_meteor(predictions=[pred] * len(references), references=references)
        semantic_score = compute_semantic_equivalence(predictions=[pred] * len(references), references=references)
        final_scores = [geometric_mean(score1, score2) for score1, score2 in zip(token_score, semantic_score)]
        best_score = max(final_scores)
        if best_score > threshold:
            sel_ref_idx = final_scores.index(best_score)
            correct_num += 1
            references[sel_ref_idx] = "xxxxxxxxxxxxxxx"
    return cal_f1(correct_num, pred_num, ref_num)
    

def eval_statement_parsing(statement_prediction: List[str], statement_references: List[str], threshold: float):
    statement_prediction = [pred.lower() for pred in statement_prediction]
    statement_references = [ref.lower() for ref in statement_references]
    correct_num = 0.0
    correct_statements_idx = []
    statement_idx = 0
    pred_num = len(statement_prediction)
    ref_num = len(statement_references)
    for pred in statement_prediction:
        token_score = compute_meteor(predictions=[pred] * len(statement_references), references=statement_references)
        semantic_score = compute_semantic_equivalence(predictions=[pred] * len(statement_references), references=statement_references)
        final_scores = [geometric_mean(score1, score2) for score1, score2 in zip(token_score, semantic_score)]
        best_score = max(final_scores)
        if best_score > threshold:
            sel_ref_idx = final_scores.index(best_score)
            correct_statements_idx.append((statement_idx, sel_ref_idx))
            correct_num += 1
            statement_references[sel_ref_idx] = "xxxxxxxxxxxxxxx"
        statement_idx += 1
    return cal_f1(correct_num, pred_num, ref_num), correct_statements_idx


def eval_statement_evidence_pair(evidence_predictions: List[str], evidence_references: List[str], correct_statements_idx: List[set], threshold: float):
    evidence_predictions = [pred.lower() for pred in evidence_predictions]
    evidence_references = [ref.lower() for ref in evidence_references]
    correct_num = 0.0
    correct_pair_idx = []
    for idx in correct_statements_idx:
        statement_idx, reference_idx = idx
        evidence_prediction = evidence_predictions[statement_idx]
        evidence_reference = evidence_references[reference_idx]
        token_score = compute_meteor(predictions=[evidence_prediction], references=[evidence_reference])[0]
        semantic_score = compute_semantic_equivalence(predictions=evidence_prediction, references=evidence_reference)[0]
        final_scores = geometric_mean(token_score, semantic_score)
        if final_scores > threshold:
            correct_num += 1
            correct_pair_idx.append((statement_idx, reference_idx))
    # breakpoint()
    return cal_f1(correct_num, len(evidence_predictions), len(evidence_references)), correct_pair_idx

def eval_reasoning(predictions: List[bool], references: List[bool], correct_pair_idx: List[set]):
    correct_num = 0.0
    for idx in correct_pair_idx:
        pair_idx, references_idx = idx
        res_prediction = predictions[pair_idx]
        res_reference = references[references_idx]
        if res_prediction == res_reference:
            correct_num += 1
    return cal_f1(correct_num, len(predictions), len(references))


def evaluate(args: argparse.Namespace) -> Dict[str, float]:
    """Evaluate the correlation between the predited pair-wise
    (statements, evidence) and the golden (statements, evidence)
    pairs by similarity-based metrics.
    """

    def load_data(args) -> Tuple[List[Any]]:
        with open(args.prediction, "r") as f_pred, open(args.reference, "r") as f_ref:
            predictions = json.load(f_pred)
            references = json.load(f_ref)

        # make sure the predictions and references have the same ids
        pred_ids = [pred["id"] for pred in predictions]
        references = [ref for ref in references if ref["id"] in pred_ids]

        if args.sample_rate < 1.0:
            predictions = predictions[: int(len(predictions) * args.sample_rate)]
            references = references[: int(len(references) * args.sample_rate)]

        return predictions, references

    # load data
    predictions, references = load_data(args)

    # build the indexes from id to prediction or reference
    id2prediction = {pred["id"]: pred for pred in predictions}
    id2reference = {ref["id"]: ref for ref in references}
    assert (
        id2prediction.keys()
        == id2reference.keys()
        == (id2prediction.keys() & id2reference.keys())
    ), "The prediction and reference files should have the same ids."

    # compute similarity for each pair of instances through iterating predictions
    total_question_f1 = 0.0
    total_statement_f1 = 0.0
    total_relation_f1 = 0.0
    total_reasoning_f1 = 0.0

    with Progress() as progress:
        console.print(f"Total number of predictions: {len(predictions)}")
        task = progress.add_task("[cyan] Evaluating ...", total=len(predictions))
        for idx in range(len(predictions)):
            
            prediction = predictions[idx]
            pred_id = prediction["id"]
            reference = id2reference[pred_id]
            ref_id = reference["id"]

            pred_question_parsing = prediction["question_parsing"]
            ref_question_parsing = reference["question_parsing"]
            metrics_question_parsing = eval_question_parsing(pred_question_parsing, ref_question_parsing, args.question_threshold)
            total_question_f1 += metrics_question_parsing

            pred_cot_parsing = prediction["cot_parsing"]  # List[Dict[str, str]]
            ref_cot_parsing = reference["cot_parsing"]  # List[Dict[str, str]]


            pred_statements = [item["statement"] for item in pred_cot_parsing]
            pred_evidences = [item["evidence"] for item in pred_cot_parsing]
            pred_verifications = [
                True if item["Verification"] == "true" or item["Verification"] == "True" else False
                for item in pred_cot_parsing
            ]

            ref_statements = [item["statement"] for item in ref_cot_parsing]
            ref_evidences = [item["evidence"] for item in ref_cot_parsing]
            ref_verifications = [
                True if item["Verification"] == "true" or item["Verification"] == "True" else False
                for item in ref_cot_parsing
            ]
            assert (
                len(ref_statements) == len(ref_evidences) == len(ref_verifications)
            )

            
            metrics_statement, correct_statements_idx = eval_statement_parsing(pred_statements, ref_statements, args.statement_threshold)
            metrics_relation, correct_pair_idx = eval_statement_evidence_pair(pred_evidences, ref_evidences, correct_statements_idx, args.relation_threshold)
            metrics_reasoning = eval_reasoning(pred_verifications, ref_verifications, correct_pair_idx)

            total_statement_f1 += metrics_statement
            total_relation_f1 += metrics_relation
            total_reasoning_f1 += metrics_reasoning
            
            

            progress.update(
                task,
                advance=1,
            )

            
            console.rule()

        avg_question_f1 = round(total_question_f1 / len(predictions), 4)
        avg_statement_f1 = round(total_statement_f1 / len(predictions), 4)
        avg_relation_f1 = round(total_relation_f1 / len(predictions), 4)
        avg_reasoning_f1 = round(total_reasoning_f1 / len(predictions), 4)
      

       
        table = Table(title="Evaluation Results", header_style="Magenta")
        table.add_column("Metric")
        table.add_column("Value")
        table.add_row("Question_Macro_F1", f"{avg_question_f1}")
        table.add_row("Statement_Macro_F1", f"{avg_statement_f1}")
        table.add_row("Statement_Evidence_Macro_F1", f"{avg_relation_f1}")
        table.add_row("Reasoning_F1", f"{avg_reasoning_f1}")
    console.print(table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction", type=str, help="Path to the predictions file.")
    parser.add_argument("--reference", type=str, help="Path to the reference file.")
    # parser.add_argument("--output", type=str, help="Path to the output file.")
    parser.add_argument(
        "--sample_rate",
        type=float,
        default=1.0,
        help="Sample rate to evaluate the predictions.",
    )
    parser.add_argument(
        "--question_threshold",
        type=float,
        default=0.1,
        help="Threshold to filter the similarity.",
    )
    parser.add_argument(
        "--statement_threshold",
        type=float,
        default=0.1,
        help="Threshold to filter the similarity.",
    )
    parser.add_argument(
        "--relation_threshold",
        type=float,
        default=0.1,
        help="Threshold to filter the similarity.",
    )
    args = parser.parse_args()

    
    model = AutoModelForSequenceClassification.from_pretrained(
        "cross-encoder/nli-deberta-v3-base"
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        "sileod/deberta-v3-large-tasksource-nli"
    )

    evaluate(args)

