# Reference: https://shwinda.medium.com/build-a-full-stack-llm-application-with-openai-flask-react-and-pinecone-part-1-f3844429a5ef
from datetime import datetime as dt
import json
from flask import Blueprint, app, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from service.llm_service import OpenAIEngine, GoogleVertexAIEngine, AnthropicEngine, MistralAIEngine
from utils.sampling import sample_dataset
from service.eval_service import evaluate_llm_responses_SQuAD, evaluate_llm_responses_truthfulqa
from utils.db_postgresql import db, EvaluationResult, ExperimentResult
from collections import defaultdict

api_blueprint = Blueprint('api', __name__)

engines = {
    "openai": OpenAIEngine(),
    "googlevertex": GoogleVertexAIEngine(),
    "mistral": MistralAIEngine(),
    "anthropic": AnthropicEngine()
}

@api_blueprint.route('/query', methods=['POST'])
def query_llms():
    data = request.get_json()
    query = data.get("query")
    selected_models = data.get("models")
    if not query:
        return jsonify({"error": "No query provided."}), 400
    if not selected_models:
        return jsonify({"error": "No models selected."}), 400
    responses = {}
    for model_name in selected_models:
        engine = engines.get(model_name)
        if engine:
            response = engine.send(query)
            responses[model_name] = response.text
        else:
            responses[model_name] = f"Error: Model '{model_name}' not found."
    return jsonify(responses)


@api_blueprint.route('/process_dataset', methods=['POST'])
def process_dataset():
    data = request.get_json()
    selected_model = data.get("selectedModel")
    selected_dataset = data.get("selectedDataset")
    selected_sampling = data.get("selectedSampling")
    
    if not selected_model:
        app.logger.error("No model selected.")
        return jsonify({"error": "No model selected."}), 400
    if not selected_dataset:
        app.logger.error("No dataset selected.")
        return jsonify({"error": "No dataset selected."}), 400
    if not selected_sampling:
        app.logger.error("No sampling size selected.")
        return jsonify({"error": "No sampling size selected."}), 400

    try:
        # Generate an experiment ID
        timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
        experiment_id = f"{selected_dataset}_{selected_sampling}_{selected_model}_{timestamp}"

        # Sample the dataset (either SQuAD or TruthfulQA)
        sampled_data = sample_dataset(selected_dataset, selected_sampling)

        if selected_dataset.lower() == "truthfulqa":
            # TruthfulQA processing: No context, only questions and additional scores
            qa_pairs = []
            for qa in sampled_data:
                try:
                    correct_answers = qa.get('correct_answers', []) 
                    incorrect_answers = qa.get('incorrect_answers', []) 
                except ValueError as e:
                    app.logger.error(f"Error converting scores for QA ID {qa.get('id')}: {e}")
                    return jsonify({"error": f"Invalid score values for QA ID {qa.get('id')}. {e}"}), 400
                qa_pairs.append({
                    "id": qa["id"],
                    "question": qa['question'],
                    "answer": qa['answer'],
                    "correct_answers": correct_answers,
                    "incorrect_answers": incorrect_answers,
                    "category": qa.get('category', ''),
                    "source": qa.get('source', '')
                })
        else:
            # SQuAD processing: Has context, question, and answers
            qa_pairs = [{
                "id": qa["id"],
                "context": qa.get('context', 'N/A'),
                "question": qa['question'],
                "answer": qa['answer']
            } for qa in sampled_data]

        # Send questions to the selected LLM
        engine = engines.get(selected_model)
        if not engine:
            return jsonify({"error": f"Model '{selected_model}' not found."}), 400

        responses = []
        for qa in qa_pairs:
            if selected_dataset.lower() == "truthfulqa":
                # No context for TruthfulQA, just question
                response = engine.send(f"Question: {qa['question']}")
            else:
                response = engine.send(f"Question: {qa['question']}\nContext: {qa['context']}")
            responses.append({
                "id": qa["id"],
                "experiment_id": experiment_id,
                "question": qa["question"],
                "correct_answer": qa["answer"],
                "llm_response": response.text
            })

        # Save the responses to a JSON file
        responses_output_file = f'./dataset/{selected_dataset}_llm_responses.json'
        with open(responses_output_file, 'w') as f:
            json.dump(responses, f, indent=2)
        print(f"Responses saved to {responses_output_file}")

        # Evaluate the responses and save results locally
        evaluation_output_file = f'./dataset/{selected_dataset}_evaluation_results.json'
        print(f"Evaluating responses and saving to {evaluation_output_file}")

        # Different evaluation for TruthfulQA and SQuAD
        if selected_dataset.lower() == "truthfulqa":
            result_data = evaluate_llm_responses_truthfulqa(qa_pairs, [resp['llm_response'] for resp in responses], evaluation_output_file, experiment_id)
        else:
            result_data = evaluate_llm_responses_SQuAD(qa_pairs, [resp['llm_response'] for resp in responses], evaluation_output_file, experiment_id)

        evaluation_results = result_data["evaluation_results"]
        average_f1_score = result_data["average_metrics"].get("average_f1_score", 0)
        average_bleu_score = result_data["average_metrics"].get("average_bleu_score", 0)
        average_rouge_score = result_data["average_metrics"].get("average_rouge_score", 0)

        # Save evaluation results to the database
        for eval_result in evaluation_results:
            result_entry = EvaluationResult(
                experiment_id=experiment_id,
                question=eval_result["question"],
                context=eval_result.get("context", None),
                correct_answer=eval_result["correct_answer"],
                llm_response=eval_result["llm_response"],
                f1_score=eval_result.get("f1_score"),
                bleu_score=eval_result.get("bleu_score"),
                rouge_score=eval_result.get("rouge_score")
            )
            db.session.add(result_entry)
        print(f"Saved {len(evaluation_results)} evaluation results to the database.")

        # Save experiment summary to ExperimentResult table
        experiment_entry = ExperimentResult(
            experiment_id=experiment_id,
            average_f1_score=average_f1_score,
            average_bleu_score=average_bleu_score,
            average_rouge_score=average_rouge_score
        )
        db.session.add(experiment_entry)
        db.session.commit()
        print(f"Experiment {experiment_id} saved with average F1 score: {average_f1_score}, average BLEU score: {average_bleu_score}, and average ROUGE score: {average_rouge_score}.")

        return jsonify({"evaluation_results": evaluation_results, "experiment_id": experiment_id})

    except ValueError as e:
        app.logger.error(f"Error processing dataset: {e}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        app.logger.error(f"General error: {e}")
        return jsonify({"error": str(e)}), 500

@api_blueprint.route('/get_evaluation_results', methods=['GET'])
def get_evaluation_results():
    try:
        results = EvaluationResult.query.all()
        response_data = []

        for result in results:
            # Prepare common response data
            result_entry = {
                "id": result.id,
                "experiment_id": result.experiment_id,
                "question": result.question,
                "correct_answer": result.correct_answer,
                "llm_response": result.llm_response,
            }
            # Check if dataset is TruthfulQA and append additional fields
            if 'TruthfulQA' in result.experiment_id:
                result_entry.update({
                    "bleu_score": result.bleu_score,
                    "rouge_score": result.rouge_score
                })
            else:
                # For SQuAD dataset
                result_entry.update({
                    "context": result.context,
                    "f1_score": result.f1_score,
                    "bleu_score": result.bleu_score
                })
            response_data.append(result_entry)
        return jsonify(response_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Use in Leaderboard
@api_blueprint.route('/get_model_averages', methods=['GET'])
def get_model_averages():
    try:
        print("Endpoint '/api/get_model_averages' was hit")  
        dataset = request.args.get('dataset', default='SQuAD', type=str)
        model_name = request.args.get('model', default='', type=str)
        print(f"Dataset received: {dataset}, Model received: {model_name}")

        # Assuming EvaluationResult is a model from SQLAlchemy
        query = EvaluationResult.query.filter(EvaluationResult.experiment_id.contains(dataset))
        if model_name:
            query = query.filter(EvaluationResult.experiment_id.contains(model_name))

        results = query.all()
        print(f"Number of results fetched: {len(results)}")
        model_scores = defaultdict(lambda: {"total_f1": 0, "total_bleu": 0, "total_rouge": 0, "count": 0})

        for result in results:
            # Print the result to see available attributes
            print(vars(result))  
            model_name_extracted = result.experiment_id.split('_')[2]
            if 'TruthfulQA' in result.experiment_id:
                # For TruthfulQA, use BLEU and ROUGE scores
                model_scores[model_name_extracted]["total_bleu"] += result.bleu_score or 0
                model_scores[model_name_extracted]["total_rouge"] += result.rouge_score or 0
            else:
                # For SQuAD, use f1_score and bleu_score
                model_scores[model_name_extracted]["total_f1"] += result.f1_score or 0
                model_scores[model_name_extracted]["total_bleu"] += result.bleu_score or 0
            model_scores[model_name_extracted]["count"] += 1

        response_data = []
        for model, data in model_scores.items():
            if 'TruthfulQA' in dataset:
                response_data.append({
                    "model": model,
                    "average_bleu_score": data["total_bleu"] / data["count"] if data["count"] > 0 else 0,
                    "average_rouge_score": data["total_rouge"] / data["count"] if data["count"] > 0 else 0,
                })
            else:
                response_data.append({
                    "model": model,
                    "average_f1_score": data["total_f1"] / data["count"] if data["count"] > 0 else 0,
                    "average_bleu_score": data["total_bleu"] / data["count"] if data["count"] > 0 else 0,
                })

        response_data = sorted(response_data, key=lambda x: x.get('average_f1_score', 0), reverse=True)
        print("Response data:", response_data)
        return jsonify(response_data)
    except Exception as e:
        print(f"General Error: {str(e)}")
        return jsonify({"error": str(e)}), 500
