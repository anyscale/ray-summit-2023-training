import openai
import re

def evaluate_retrieval(llama_index_retriever, queries, golden_sources):
    results = []

    for query, expected_source in zip(queries, golden_sources):
        retrieved_nodes = llama_index_retriever.retrieve(query)
        retrieved_sources = [node.metadata['source'] for node in retrieved_nodes]
        
        # If our label does not include a section, then any sections on the page should be considered a hit.
        if "#" not in expected_source:
            retrieved_sources = [source.split("#")[0] for source in retrieved_sources]
        
        if expected_source in retrieved_sources:
            is_hit = True
            score = retrieved_nodes[retrieved_sources.index(expected_source)].score
        else:
            is_hit = False
            score = 0.0
        
        result = {
            "is_hit": is_hit,
            "score": score,
            "retrieved": retrieved_sources,
            "expected": expected_source,
            "query": query,
        }
        results.append(result)
    return results

def _generate_response(llm_name, 
                      temperature, 
                      system_content, 
                      user_content):
    
    response = openai.ChatCompletion.create(
        model=llm_name,
        temperature=temperature,
        messages=[
                {"role": "system", "content": system_content},
                {"role": "assistant", "content": ""},
                {"role": "user", "content": user_content},
            ],)
    return response["choices"][-1]["message"]["content"]

def _parse_response(response):
    # Define regular expressions for extracting values
    score_pattern = r'"score"\s*:\s*([0-9]+)'
    reasoning_pattern = r'"reasoning"\s*:\s*"([^"]*)"'

    # Extract values using regular expressions
    score_match = re.search(score_pattern, response)
    reasoning_match = re.search(reasoning_pattern, response)

    # Convert
    if score_match and reasoning_match:
        score = float(score_match.group(1))
        reasoning = reasoning_match.group(1)
        return {"score": score, "reasoning": reasoning}

    return {"score": "", "reasoning": ""}

def evaluate_e2e(llama_index_query_engine, queries, golden_responses):
    generated_responses_str = []

    for query in queries:
        response = llama_index_query_engine.query(query)
        generated_responses_str.append(response.response)
    
    
    # Evaluation prompt
    system_content = """
        "You are given a query, a reference answer, and a candidate answer.
        You must {score} the candidate answer between 1 and 5 on how well it answers the query, 
        using the reference answer as the golden truth.
        You must return your response in a line with only the score.
        Do not add any more details.
        On a separate line provide your {reasoning} for the score as well.
        Return your response following the exact format outlined below.
        All of this must be in a valid JSON format.
        
        {"score": score,
        "reasoning": reasoning}
        """

    evaluation_scores = []
    llm_name = "gpt-4"
    max_context_length = 8192


    for query, generated_answer, golden_answer in zip(queries, generated_responses_str, golden_responses):
        
        context_length = max_context_length - len(system_content)
        user_content = f"The query is {query}, the reference answer is {golden_answer}, and the candidate answer is {generated_answer}"[:context_length]
        
        response = _generate_response(llm_name, temperature=0.0, system_content=system_content, user_content=user_content)
        parsed_response = _parse_response(response)
        parsed_response["query"] = query
        parsed_response["generated_response"] = generated_answer
        parsed_response["golden_response"] = golden_answer
        evaluation_scores.append(parsed_response)
    
    return evaluation_scores