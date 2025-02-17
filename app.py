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

    def _format_response(self, response: Response) -> str:
        """Format the model response with confidence information"""
        confidence_color = self._get_confidence_color(response.confidence)
        confidence_info = f"<span style='color: {confidence_color}'>[Confidence: {response.confidence:.2f}]</span>"
        
        response_type = f"<span style='color: gray'>[{response.type.value}]</span>"
        
        return f"{response.text} {confidence_info} {response_type}"

    def _get_confidence_color(self, confidence: float) -> str:
        """Return color based on confidence level"""
        if confidence >= 0.8:
            return "green"
        elif confidence >= 0.5:
            return "orange"
        return "red"

    def chat(self, message: str, history: List[Tuple[str, str]]) -> Tuple[str, List[Tuple[str, str]]]:
        """Process a chat message and update history"""
        if not message.strip():
            return "", history
            
        # Get response from model
        response = self.model.process_query(message)
        
        # Format response with confidence and type information
        formatted_response = self._format_response(response)
        
        # Update history
        history.append((message, formatted_response))
        if len(history) > self.max_history:
            history = history[-self.max_history:]
            
        self.history = history
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
        """Get formatted memory statistics"""
        stats = self.model.get_memory_stats()
        
        return f"""
        Memory Statistics:
        - Patterns: {stats.pattern_count}
        - Unique Tokens: {stats.token_count}
        - Vector Memory: {stats.vector_memory_mb:.2f}MB
        
        Confidence Distribution:
        {json.dumps(stats.confidence_histogram, indent=2)}
        
        Recent Usage:
        {json.dumps([(p, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t))) 
                     for p, t in stats.recent_usage], indent=2)}
        """

def create_chatbot_interface():
    """Create and launch the Gradio interface"""
    chatbot = ChatbotInterface()
    
    # Create the interface
    with gr.Blocks(css="footer {display: none !important}") as interface:
        gr.Markdown("""
        # ZeroShotLM Chatbot
        This chatbot learns from interactions and uses context-aware response generation.
        """)
        
        with gr.Row():
            with gr.Column(scale=4):
                # Main chat interface
                chatbot_component = gr.Chatbot(
                    show_label=False,
                    height=600
                )
                
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
            chatbot.chat,
            inputs=[message_input, chatbot_component],
            outputs=[message_input, chatbot_component]
        )
        
        message_input.submit(
            chatbot.chat,
            inputs=[message_input, chatbot_component],
            outputs=[message_input, chatbot_component]
        )
        
        clear_btn.click(
            chatbot.reset_chat,
            outputs=[message_input, chatbot_component]
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