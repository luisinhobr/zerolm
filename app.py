import gradio as gr
import numpy as np
from typing import List, Tuple, Dict
import time
import json
from pathlib import Path
from collections import deque
import logging

# Import our ZeroShotLM implementation
from zerolm import ZeroShotLM, Response, ResponseType

class ChatbotInterface:
    def __init__(self):
        # Initialize the ZeroShotLM model with common settings
        self.model = ZeroShotLM(
            use_sparse_vectors=True,
            vector_dim=100,
            min_confidence=0.3,
            temporal_weight=0.7,
            context_window=5,
            language="en",
            max_patterns=10000,
            learning_rate=0.1
        )
        
        self.max_history = 50
        self.history = deque(maxlen=self.max_history * 2)
        
        # Path for saving/loading model state
        self.save_path = Path("chatbot_state")
        self.save_path.mkdir(exist_ok=True)
        
        # Try to load existing model state
        self._load_state()

        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

    def _format_response(self, response: Response) -> dict:
        """Formata a resposta no formato correto para o Gradio"""
        return {
            "role": "assistant",
            "content": f"{response.text} [Confiança: {response.confidence:.2f}]"
        }

    def chat(self, message: str, history: List[dict]) -> Tuple[str, List[dict]]:
        """
        Process a chat message and update history
        
        Args:
            message: User input text
            history: List of message dictionaries
            
        Returns:
            Tuple of (empty string, updated history)
        """
        # Convert input list to deque temporarily
        temp_history = deque(history, maxlen=self.max_history * 2)
        
        if not message.strip():
            return "", list(temp_history)
            
        # Obtém resposta do modelo
        response = self.model.process_query(message)
        
        # Formata no novo padrão
        formatted_response = self._format_response(response)
        
        # Atualiza histórico com dicionários
        temp_history.append({"role": "user", "content": message})
        temp_history.append(formatted_response)
        
        # Mantém apenas o histórico máximo
        if len(temp_history) > self.max_history * 2:  # Multiplica por 2 (user + assistant)
            temp_history = temp_history[-self.max_history * 2:]
            
        # Return as list for Gradio compatibility
        return "", list(temp_history)

    def learn_response(self, message: str, correct_response: str) -> str:
        """Teach the model a new response"""
        if not message.strip() or not correct_response.strip():
            return "Both message and response must be provided"
            
        success = self.model.learn(message, correct_response)
        
        if success:
            self._save_state()  # Save after learning
            return "Successfully learned new response!"
        return "Failed to learn response. Please try again."

    def reset_chat(self) -> Tuple[str, List[Tuple[str, str]]]:
        """Reset the chat history"""
        self.history.clear()
        return "", []

    def _save_state(self) -> str:
        """Save model state with detailed error handling"""
        try:
            save_file = self.save_path / "model_state.pkl"
            self.model.save(str(save_file))
            msg = f"Saved successfully at {time.strftime('%Y-%m-%d %H:%M:%S')}"
            self.logger.info(msg)
            return msg
        except PermissionError as pe:
            self.logger.error(f"Permission error: {str(pe)}")
            return "Error: File permission denied"
        except Exception as e:
            self.logger.error(f"Save failed: {str(e)}")
            return f"Error saving state: {str(e)}"

    def _load_state(self) -> str:
        """Load model state from disk"""
        try:
            save_file = self.save_path / "model_state.pkl"
            if save_file.exists():
                self.model.load(str(save_file))
                return "Model state loaded successfully"
            return "No saved state found"
        except Exception as e:
            return f"Error loading model state: {str(e)}"

    def get_memory_stats(self) -> str:
        """Return formatted memory statistics"""
        return self.model.get_memory_stats()

def create_chatbot_interface():
    """Cria interface com configuração correta de mensagens"""
    chatbot = ChatbotInterface()
    
    with gr.Blocks() as interface:
        # Configuração da interface corrigida
        gr.Markdown("# Chatbot ZeroShotLM")
        
        chat_messages = gr.Chatbot(
            label="Conversa",
            type="messages",
            avatar_images=("user.png", "bot.png")
        )
        
        with gr.Row():
            with gr.Column(scale=4):
                # Main chat interface
                chatbot_component = chat_messages
                
                with gr.Row():
                    message_input = gr.Textbox(
                        show_label=False,
                        placeholder="Type your message here...",
                        scale=8
                    )
                    submit_btn = gr.Button("Send", scale=1)
                    clear_btn = gr.Button("Clear", scale=1)
            
            with gr.Column(scale=2):
                # Learning interface
                gr.Markdown("### Teach the Chatbot")
                teach_message = gr.Textbox(label="Message to learn")
                teach_response = gr.Textbox(label="Correct response")
                teach_btn = gr.Button("Teach")
                learn_output = gr.Textbox(label="Learning Status", interactive=False)
                
                # Stats display with proper configuration
                stats_output = gr.Textbox(
                    label="Memory Statistics",
                    interactive=False,
                    value=chatbot.get_memory_stats(),
                    every=30  # Automatic updates every 30 seconds
                )
                refresh_stats_btn = gr.Button("Refresh Stats")
        
        # Set up event handlers
        submit_btn.click(
            lambda msg, hist: chatbot.chat(msg, hist),
            inputs=[message_input, chat_messages],
            outputs=[message_input, chat_messages]
        )
        
        message_input.submit(
            chatbot.chat,
            inputs=[message_input, chat_messages],
            outputs=[message_input, chat_messages]
        )
        
        clear_btn.click(
            chatbot.reset_chat,
            outputs=[message_input, chat_messages]
        )
        
        teach_btn.click(
            chatbot.learn_response,
            inputs=[teach_message, teach_response],
            outputs=[learn_output]
        )
        
        refresh_stats_btn.click(
            lambda: chatbot.get_memory_stats(),
            outputs=[stats_output]
        )
    
    return interface

if __name__ == "__main__":
    interface = create_chatbot_interface()
    interface.launch(share=True)
