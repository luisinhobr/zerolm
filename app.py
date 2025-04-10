import gradio as gr
from typing import List, Tuple, Dict
import time
from pathlib import Path
from collections import deque
import logging

from zerolm.core.zero_shot_lm import ZeroShotLM
from zerolm.core.response import Response
from zerolm.types import ResponseType
from zerolm.memory import MemoryStats


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
        """Format the response for Gradio interface"""
        return {
            "role": "assistant",
            "content": f"{response.text} [Confidence: {response.confidence:.2f}]"
        }

    def chat(self, message: str, history: List[dict]) -> Tuple[str, List[dict]]:
        """Process a chat message and update history"""
        # Convert input list to deque temporarily
        temp_history = deque(history, maxlen=self.max_history * 2)
        
        if not message.strip():
            return "", list(temp_history)
            
        # Get response from model
        response = self.model.process_query(message)
        
        # Format response
        formatted_response = self._format_response(response)
        
        # Update history with dictionaries
        temp_history.append({"role": "user", "content": message})
        temp_history.append(formatted_response)
        
        # Keep only max history
        if len(temp_history) > self.max_history * 2:  # Multiply by 2 (user + assistant)
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

    def reset_chat(self) -> Tuple[str, List[dict]]:
        """Reset the chat history"""
        self.history.clear()
        return "", []
    def _save_state(self) -> str:
        """Save model state with detailed error handling"""
        try:
            save_file = self.save_path / "model_state.pkl"
            # Create backup before saving
            if save_file.exists():
                backup_file = self.save_path / "model_state.backup.pkl"
                save_file.rename(backup_file)
            
            self.model.save(str(save_file))
        
            # Verify saved file
            with open(str(save_file), 'rb') as f:
                content = f.read()
            if b'\x00' in content:
                if backup_file.exists():
                    backup_file.rename(save_file)
                raise ValueError("Generated file contains null bytes")
                
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
                # Read file in binary mode and check for null bytes
                with open(str(save_file), 'rb') as f:
                    content = f.read()
                    if b'\x00' in content:
                        self.logger.error("File contains null bytes - corrupted data")
                        return "Error: Model state file is corrupted (contains null bytes)"
                    self.model.load(str(save_file))
                    return "Model state loaded successfully"
            return "No saved state found"
        except Exception as e:
            self.logger.error(f"Load failed: {str(e)}")
            return f"Error loading model state: {str(e)}"

    def get_memory_stats(self) -> MemoryStats:
        """Return memory statistics"""
        return self.model.get_memory_stats()


def create_chatbot_interface():
    """Create interface with correct message configuration"""
    chatbot = ChatbotInterface()
    
    with gr.Blocks() as interface:
        gr.Markdown("# ZeroShotLM Chatbot")
        
        chat_messages = gr.Chatbot(
            label="Conversation",
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
                
                # Stats display
                stats_output = gr.Textbox(
                    label="Memory Statistics",
                    interactive=False,
                    value=str(chatbot.get_memory_stats()),
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
            lambda: str(chatbot.get_memory_stats()),
            outputs=[stats_output]
        )
    
        return interface
if __name__ == "__main__":
    interface = create_chatbot_interface()
    interface.launch(share=True)