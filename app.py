import gradio as gr
import numpy as np
from typing import List, Tuple
import time
import json
from pathlib import Path

# Import our ZeroShotLM implementation
from zerolm import ZeroShotLM, Response, ResponseType

class ChatbotInterface:
    def __init__(self):
        # Initialize the ZeroShotLM model with common settings
        self.model = ZeroShotLM(
            use_vectors=True,
            vector_dim=100,
            min_confidence=0.3,
            temporal_weight=0.7,
            context_window=5,
            language="en",
            max_patterns=10000,
            learning_rate=0.1
        )
        
        # Chat history storage
        self.history: List[Tuple[str, str]] = []
        self.max_history = 50
        
        # Path for saving/loading model state
        self.save_path = Path("chatbot_state")
        self.save_path.mkdir(exist_ok=True)
        
        # Try to load existing model state
        self._load_state()

    def _format_response(self, response: Response) -> dict:
        """Formata a resposta no formato correto para o Gradio"""
        return {
            "role": "assistant",
            "content": f"{response.text} [Confiança: {response.confidence:.2f}]"
        }

    def chat(self, message: str, history: list) -> tuple:
        """Processa a mensagem e atualiza o histórico corretamente"""
        if not message.strip():
            return "", history
            
        # Obtém resposta do modelo
        response = self.model.process_query(message)
        
        # Formata no novo padrão
        formatted_response = self._format_response(response)
        
        # Atualiza histórico com dicionários
        history.append({"role": "user", "content": message})
        history.append(formatted_response)
        
        # Mantém apenas o histórico máximo
        if len(history) > self.max_history * 2:  # Multiplica por 2 (user + assistant)
            history = history[-self.max_history * 2:]
            
        return "", history

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
        self.history = []
        return "", []

    def _save_state(self) -> str:
        """Save model state to disk"""
        try:
            save_file = self.save_path / "model_state.pkl"
            self.model.save(str(save_file))
            return f"Model state saved successfully at {time.strftime('%Y-%m-%d %H:%M:%S')}"
        except Exception as e:
            return f"Error saving model state: {str(e)}"

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
        """Retorna estatísticas formatadas da memória"""
        return self.model.get_memory_stats()

def create_chatbot_interface():
    """Cria interface com configuração correta de mensagens"""
    chatbot = ChatbotInterface()
    
    with gr.Blocks() as interface:
        gr.Markdown("# Chatbot ZeroShotLM")
        
        # Componente de chat configurado corretamente
        chat_messages = gr.Chatbot(
            label="Conversa",
            height=600,
            avatar_images=("assets/user.png", "assets/bot.png"),
            type="messages",
            examples=[
                {"role": "user", "content": "O que é inteligência artificial?"},
                {"role": "user", "content": "Explique machine learning"}
            ]
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
                
                # Stats display
                gr.Markdown("### Memory Statistics")
                stats_output = gr.Textbox(
                    label="Stats",
                    interactive=False,
                    value=chatbot.get_memory_stats()
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
        
        # Update stats periodically
        gr.Textbox.update(value=chatbot.get_memory_stats(), every=30)
    
    return interface

if __name__ == "__main__":
    interface = create_chatbot_interface()
    interface.launch(share=True)
