from app.core.constant import Constants
from app.core.schema import Context
from app.clients.llm_client import get_structured_data

class TranslationService:
    """
    Handles the business logic for translation, including context management
    and calling the LLM client.
    """
    def __init__(self):
        self.context_history = []
        self.context_window = Constants.CONTEXT_WINDOW

    def _prepare_user_prompt(self, japanese_text: str, size: int) -> str:
        """
        Prepares the user content part of the prompt, including context.
        """
        context_str = ""
        if self.context_history:
            history_items = []
            for item in self.context_history[-self.context_window:]:
                history_items.append(
                    Constants.CONTEXT_HISTORY_ITEM_TEMPLATE.format(
                        japanese=item['japanese'],
                        english=item['english']
                    )
                )
            context_str = Constants.CONTEXT_SECTION_TEMPLATE.format(
                context_history="".join(history_items)
            )

        # This is a simplified way to pass the user-specific parts to the llm_client
        # The main template is now in the system message within the client.
        user_prompt = f"""
        {context_str}
        
        Japanese Text (must translate to exactly {size} lines):
        ---
        {japanese_text}
        ---
        """
        return user_prompt

    async def translate(self, japanese_text: str, size: int) -> str:
        """
        Translates a chunk of Japanese text, ensuring line count consistency.
        """
        user_prompt = self._prepare_user_prompt(japanese_text, size)
        
        translated_text = ""
        output_lines = 0
        
        # Loop until the output line count matches the input size
        while output_lines != size:
            # The llm_client now handles the main prompt template
            response_data = await get_structured_data(user_prompt, Context)
            
            translated_text = "\n".join(response_data["translated_outputs"])
            output_lines = len(translated_text.splitlines())
            
            if output_lines != size:
                print(f"Warning: Line count mismatch. Expected {size}, got {output_lines}. Retrying...")

        # Update context history
        self.context_history.append({
            'japanese': japanese_text,
            'english': translated_text
        })
        
        # Trim history
        if len(self.context_history) > self.context_window:
            self.context_history = self.context_history[-self.context_window:]
            
        return translated_text
