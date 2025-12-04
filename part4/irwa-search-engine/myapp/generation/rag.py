import os
from groq import Groq
from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env


class RAGGenerator:

    PROMPT_TEMPLATE = """
        You are an expert product advisor helping users choose the best option from retrieved e-commerce products.

        ## Instructions:
        1. Explain why this product matches the user's request. **Crucially, use product attributes like Price, Rating, and Discount to justify your recommendation.**
        2. Present the recommendation clearly in this format:
        - Best Product: [Product PID] [Product Name]
        - Why: [Explain in plain language why this product is the best fit, referring to specific attributes like price, features, quality, or fit to user's needs.]
        3. If there is another product that could also work, mention it briefly as an alternative.
        4. If no product is a good fit, return ONLY this exact phrase:
        "There are no good products that fit the request based on the retrieved results."

        ## Best Retrieved Product (PID, Title, Price, Rating, Discount):
        {best_product}

        #Alternative:
        {alt_product}
        ## User Request:
        {user_query}

        ## Output Format:
        - Best Product: ...
        - Why: ...
        - Alternative (optional): ...
    """

    def generate_response(self, user_query: str, retrieved_results: list, top_N: int = 20) -> str:
        """
        Generate a response using the retrieved search results. 
        Returns:
            str: The generated suggestion or a default error message.
        """
        DEFAULT_ANSWER = "RAG is not available. Check your credentials (.env file) or account limits."
        
        # IMPROVEMENT 1: Check for empty results before calling the LLM
        if not retrieved_results:
             return "There are no good products that fit the request based on the retrieved results."
        
        try:
            client = Groq(
                api_key=os.environ.get("GROQ_API_KEY"),
            )
            model_name = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")

            # IMPROVEMENT 2: Format the results for the prompt to include  price, rating, and discount
            formatted_results = "\n".join(
                [
                    f"- PID: {res.pid}, Title: {res.title}, Price: {res.selling_price if res.selling_price is not None else 'N/A'}, Rating: {res.average_rating if res.average_rating is not None else 'N/A'}, Discount: {res.discount if res.discount is not None else 'N/A'}%"
                    for res in retrieved_results[:top_N]
                ]
            )
            
            # Do another security check
            if not formatted_results and not retrieved_results: 
                 formatted_results = "No products retrieved or formatted for analysis."


            best = formatted_results.split("\n")[0] if formatted_results else "N/A"
            alt = formatted_results.split("\n")[1] if len(formatted_results.split("\n")) > 1 else "N/A"

            prompt = self.PROMPT_TEMPLATE.format(
                best_product=best,
                alt_product=alt,
                user_query=user_query
)

            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model=model_name,
            )

            generation = chat_completion.choices[0].message.content
            return generation
        except Exception as e:
            print(f"Error during RAG generation: {e}")
            return DEFAULT_ANSWER