import os
import json
import random
from tqdm import tqdm
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

class ModelTrainer:
    """Class to generate synthetic QA data and evaluate Flash mode performance."""
    
    def __init__(self, rag_system):
        """Initialize the trainer with the RAG system."""
        self.rag = rag_system
        self.training_data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "training_data.json")
        self.eval_results_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "eval_results.json")
        
        # Create data directory if it doesn't exist
        os.makedirs(os.path.join(os.path.dirname(os.path.dirname(__file__)), "data"), exist_ok=True)
    
    def create_synthetic_dataset(self, num_examples=50) -> List[Dict[str, str]]:
        """
        Create a synthetic QA dataset from scraped content.
        
        Args:
            num_examples: Number of examples to generate
            
        Returns:
            List of dictionaries containing question, context, and expected answer
        """
        print(f"Generating synthetic QA dataset with {num_examples} examples...")
        
        # Get documents from vector database
        documents = self.rag.vectordb.get_all_documents()
        if not documents:
            print("No documents found in vector database. Please scrape a website first.")
            return []
        
        # Randomly sample documents if we have more than we need
        if len(documents) > num_examples:
            documents = random.sample(documents, num_examples)
        
        training_data = []
        
        # For each document, generate a question and answer
        for doc in tqdm(documents, desc="Generating QA pairs"):
            content = doc['content']
            
            # Skip if content is too short
            if len(content.split()) < 20:
                continue
                
            # Generate a question based on the content
            question_prompt = f"""
            Given the following text, generate a specific factual question that can be answered using only this text.
            The question should be about a specific fact, entity, or relationship mentioned in the text.
            
            TEXT:
            {content}
            
            QUESTION:
            """
            
            try:
                # Use the speed model to generate a question with lower temperature for specificity
                try:
                    response = self.rag.speed_llm.generate(
                        prompt=question_prompt,
                        max_tokens=50,
                        temperature=0.3
                    )
                    
                    # Extract the response text
                    if isinstance(response, dict) and 'response' in response:
                        question = response['response'].strip()
                    elif isinstance(response, str):
                        question = response.strip()
                    else:
                        question = str(response).strip()
                        
                    # Now generate the answer to this question based on the content
                    answer_prompt = f"""
                    Given the following text and question, provide a concise and accurate answer based ONLY on the information in the text.
                    If the question cannot be answered from the text, respond with "I don't have enough information to answer this question."
                    
                    TEXT:
                    {content}
                    
                    QUESTION:
                    {question}
                    
                    ANSWER:
                    """
                    
                    # Generate answer with low temperature for accuracy using the deep model
                    response = self.rag.llm.generate(
                        prompt=answer_prompt,
                        max_tokens=100,
                        temperature=0.2
                    )
                    
                    # Extract the response text
                    if isinstance(response, dict) and 'response' in response:
                        answer = response['response'].strip()
                    elif isinstance(response, str):
                        answer = response.strip()
                    else:
                        answer = str(response).strip()
                except Exception as e:
                    # If the primary models fail, try the backup models
                    print(f"Primary models failed, trying backup: {str(e)}")
                    try:
                        # Try using another model as backup
                        response = self.rag.quick_llm.generate(
                            prompt=question_prompt,
                            max_tokens=50,
                            temperature=0.3
                        )
                        
                        if isinstance(response, dict) and 'response' in response:
                            question = response['response'].strip()
                        elif isinstance(response, str):
                            question = response.strip()
                        else:
                            question = str(response).strip()
                            
                        response = self.rag.quick_llm.generate(
                            prompt=answer_prompt,
                            max_tokens=100,
                            temperature=0.2
                        )
                        
                        if isinstance(response, dict) and 'response' in response:
                            answer = response['response'].strip()
                        elif isinstance(response, str):
                            answer = response.strip()
                        else:
                            answer = str(response).strip()
                    except Exception as e2:
                        # Both attempts failed
                        raise Exception(f"Failed to generate QA pair: {str(e2)}")
                
                # Add to training data
                training_data.append({
                    "question": question,
                    "context": content,
                    "expected_answer": answer
                })
                
            except Exception as e:
                print(f"Error generating QA pair: {str(e)}")
                continue
        
        # Save the training data
        with open(self.training_data_path, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, indent=2, ensure_ascii=False)
            
        print(f"Generated {len(training_data)} QA pairs. Saved to {self.training_data_path}")
        return training_data
    
    def load_training_data(self) -> List[Dict[str, str]]:
        """Load existing training data if available."""
        if os.path.exists(self.training_data_path):
            with open(self.training_data_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []
    
    def evaluate_flash_mode(self, training_data=None, num_examples=20) -> Dict[str, Any]:
        """
        Evaluate Flash mode performance on synthetic QA dataset.
        
        Args:
            training_data: Optional list of QA pairs to evaluate on
            num_examples: Number of examples to evaluate if training_data not provided
            
        Returns:
            Dictionary containing evaluation metrics
        """
        # Load or create training data
        if training_data is None:
            training_data = self.load_training_data()
            if not training_data:
                print("No training data found. Generating new data...")
                training_data = self.create_synthetic_dataset(num_examples)
        
        # Sample if we have more than needed
        if len(training_data) > num_examples:
            eval_data = random.sample(training_data, num_examples)
        else:
            eval_data = training_data
            
        results = {
            "total_questions": len(eval_data),
            "questions": [],
            "metrics": {
                "accuracy": 0,
                "relevance_score": 0,
                "answer_completeness": 0
            }
        }
        
        print(f"Evaluating Flash mode on {len(eval_data)} questions...")
        
        for idx, item in enumerate(tqdm(eval_data, desc="Evaluating questions")):
            question = item["question"]
            expected_answer = item["expected_answer"]
            
            # Get Flash mode response
            response = self.rag.query(question, mode="flash")
            flash_response = response["answer"] if isinstance(response, dict) and "answer" in response else str(response)
            
            # Calculate relevance score by comparing Flash response with expected answer
            # This is a crude similarity measure - would be better with embedding similarity
            relevance_score = self._calculate_text_similarity(flash_response, expected_answer)
            
            # Check if key elements from expected answer are in the flash response
            key_elements = self._extract_key_elements(expected_answer)
            elements_present = sum(1 for elem in key_elements if elem.lower() in flash_response.lower())
            completeness = elements_present / max(1, len(key_elements))
            
            # Store results for this question
            question_result = {
                "id": idx,
                "question": question,
                "expected_answer": expected_answer,
                "flash_response": flash_response,
                "relevance_score": relevance_score,
                "completeness": completeness
            }
            
            results["questions"].append(question_result)
            
            # Update overall metrics
            results["metrics"]["relevance_score"] += relevance_score
            results["metrics"]["answer_completeness"] += completeness
        
        # Calculate averages
        n = len(eval_data)
        results["metrics"]["relevance_score"] /= n
        results["metrics"]["answer_completeness"] /= n
        results["metrics"]["accuracy"] = results["metrics"]["relevance_score"] * 0.7 + results["metrics"]["answer_completeness"] * 0.3
        
        # Save evaluation results
        with open(self.eval_results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"Evaluation complete. Results saved to {self.eval_results_path}")
        print(f"Overall accuracy: {results['metrics']['accuracy']:.2f}")
        print(f"Relevance score: {results['metrics']['relevance_score']:.2f}")
        print(f"Answer completeness: {results['metrics']['answer_completeness']:.2f}")
        
        return results
    
    def generate_text(self, prompt: str, max_tokens: int = 100, temperature: float = 0.3) -> str:
        """
        Helper method to generate text using the RAG system's LLM models.
        
        Args:
            prompt: The prompt to send to the LLM
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature for generation (higher = more creative)
            
        Returns:
            Generated text as a string
        """
        try:
            # Try the deep model first
            response = self.rag.llm.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            # Extract the response text
            if isinstance(response, dict) and 'response' in response:
                return response['response'].strip()
            elif isinstance(response, str):
                return response.strip()
            else:
                return str(response).strip()
        except Exception as e:
            # Try fallback to quick model
            print(f"Deep model failed, falling back to quick model: {str(e)}")
            try:
                response = self.rag.quick_llm.generate(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                
                if isinstance(response, dict) and 'response' in response:
                    return response['response'].strip()
                elif isinstance(response, str):
                    return response.strip()
                else:
                    return str(response).strip()
            except Exception as e2:
                raise Exception(f"All models failed to generate text: {str(e2)}")
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate a more robust similarity score between two texts using multiple methods.
        
        Args:
            text1: First text (typically the model response)
            text2: Second text (typically the expected answer)
            
        Returns:
            Similarity score between 0 and 1
        """
        # Normalize and clean texts
        text1_clean = text1.lower().strip()
        text2_clean = text2.lower().strip()
        
        # 1. Direct substring check - often answers are partially contained
        # This handles cases where the model's answer is shorter but correct
        if text1_clean in text2_clean or text2_clean in text1_clean:
            return 0.75  # High score but not perfect
        
        # 2. Tokenize into words
        words1 = set(text1_clean.split())
        words2 = set(text2_clean.split())
        
        # Get more meaningful words by removing stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 
                      'be', 'to', 'of', 'for', 'in', 'that', 'on', 'with', 'this', 'it', 
                      'i', 'you', 'he', 'she', 'they', 'we', 'there', 'here', 'what', 
                      'where', 'when', 'who', 'how', 'which', 'would', 'could', 'should'}
        
        content_words1 = words1 - stop_words
        content_words2 = words2 - stop_words
        
        # 3. Exact key phrase matching (2+ word phrases)
        phrases1 = self._extract_phrases(text1_clean)
        phrases2 = self._extract_phrases(text2_clean)
        
        phrase_matches = len(set(phrases1).intersection(set(phrases2)))
        
        # 4. Calculate weighted scores
        
        # Keyword overlap (Jaccard similarity)
        if not content_words1 or not content_words2:
            keyword_score = 0.0
        else:
            intersection = len(content_words1.intersection(content_words2))
            union = len(content_words1.union(content_words2))
            keyword_score = intersection / union
        
        # Phrase matching score
        phrase_score = min(1.0, phrase_matches / max(1, min(len(phrases1), len(phrases2))))
        
        # Simple length ratio score - penalizes answers that are way too short or long
        len_ratio = min(len(text1_clean), len(text2_clean)) / max(len(text1_clean), len(text2_clean))
        
        # 5. Combine scores with appropriate weights
        final_score = 0.50 * keyword_score + 0.35 * phrase_score + 0.15 * len_ratio
        
        # Boost score if a significant portion of keywords are present
        keyword_coverage = len(content_words1.intersection(content_words2)) / max(1, len(content_words2))
        if keyword_coverage > 0.6:  # If most important keywords are present
            final_score = min(1.0, final_score * 1.25)  # Boost score by up to 25%
        
        return final_score
        
    def _extract_phrases(self, text: str, min_length: int = 2, max_length: int = 3) -> List[str]:
        """
        Extract meaningful phrases (n-grams) from text.
        
        Args:
            text: Text to extract phrases from
            min_length: Minimum number of words in a phrase
            max_length: Maximum number of words in a phrase
            
        Returns:
            List of extracted phrases
        """
        words = text.split()
        phrases = []
        
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 
                      'be', 'to', 'of', 'for', 'in', 'that', 'on', 'with'}
        
        for n in range(min_length, min(max_length + 1, len(words))):
            for i in range(len(words) - n + 1):
                phrase = ' '.join(words[i:i+n])
                # Only add phrases that don't start or end with stop words
                if words[i] not in stop_words and words[i+n-1] not in stop_words:
                    phrases.append(phrase)
        
        return phrases
    
    def _extract_key_elements(self, text: str) -> List[str]:
        """
        Extract key elements (noun phrases, entities) from a text.
        This is a simplified implementation - a more sophisticated NLP approach would be better.
        
        Args:
            text: Text to extract key elements from
            
        Returns:
            List of key elements
        """
        # Split into sentences
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        # Extract noun phrases and entities (very simplistic approach)
        key_elements = []
        
        # Extract all non-stop word sequences of 2-3 words as potential key elements
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 'be', 'to', 'of', 'for', 'in', 'that', 'on', 'with'}
        words = [w for w in text.lower().split() if w not in stop_words]
        
        for i in range(len(words) - 1):
            key_elements.append(f"{words[i]} {words[i+1]}")
            
        for i in range(len(words) - 2):
            key_elements.append(f"{words[i]} {words[i+1]} {words[i+2]}")
        
        # Add important single words (proper nouns, etc.)
        for word in text.split():
            if word[0].isupper() and len(word) > 1 and word.lower() not in stop_words:
                key_elements.append(word)
        
        # Remove duplicates and return
        return list(set(key_elements))
