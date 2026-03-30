"""
Gradio interface for Customer Support Triage RL Environment.
"""

import gradio as gr
from models import TriageAction, ResetRequest
from server.customer_support_env import CustomerSupportEnv

# Global environment
env = None

def initialize_env():
    global env
    if env is None:
        env = CustomerSupportEnv()
    return env

def reset_episode(difficulty: str, seed: int = None):
    """Reset episode"""
    env = initialize_env()
    try:
        obs = env.reset(difficulty=difficulty, seed=seed if seed != 0 else None)
        
        ticket = obs.ticket_info
        if ticket:
            ticket_text = f"""
            **Ticket ID:** {ticket.ticket_id}
            **Subject:** {ticket.subject}
            **Category (Initial):** {ticket.initial_category}
            **Description:** {ticket.description}
            **Sentiment:** {ticket.customer_sentiment:.2f}
            **Word Count:** {ticket.word_count}
            **Account Age (days):** {ticket.customer_account_age}
            **Previous Tickets:** {ticket.previous_tickets_count}
            """
        else:
            ticket_text = "No ticket"
        
        return (
            ticket_text,
            f"Progress: {obs.task_progress:.1%}",
            f"Correctness: {obs.correctness_score:.2f}",
            f"Efficiency: {obs.efficiency_score:.2f}",
            f"Reward: {obs.reward:.3f}",
        )
    except Exception as e:
        return f"Error: {str(e)}", "", "", "", ""

def step_action(route: str, urgency: str, difficulty: str, priority: float):
    """Submit action"""
    env = initialize_env()
    try:
        action = TriageAction(
            route_category=route,
            urgency_assessment=urgency,
            resolution_difficulty=difficulty,
            priority_score=priority
        )
        obs = env.step(action)
        
        ticket = obs.ticket_info
        if ticket:
            ticket_text = f"""
            **Ticket ID:** {ticket.ticket_id}
            **Subject:** {ticket.subject}
            **Category (Initial):** {ticket.initial_category}
            **Description:** {ticket.description}
            **Sentiment:** {ticket.customer_sentiment:.2f}
            **Word Count:** {ticket.word_count}
            **Account Age (days):** {ticket.customer_account_age}
            **Previous Tickets:** {ticket.previous_tickets_count}
            """
        else:
            ticket_text = "Episode Complete! ✅"
        
        done_text = "✅ Episode Done!" if obs.done else "⏳ Continue..."
        
        return (
            ticket_text,
            f"Progress: {obs.task_progress:.1%}",
            f"Correctness: {obs.correctness_score:.2f}",
            f"Efficiency: {obs.efficiency_score:.2f}",
            f"Reward: {obs.reward:.3f}",
            done_text,
        )
    except Exception as e:
        return f"Error: {str(e)}", "", "", "", "", ""

# Gradio Interface
with gr.Blocks(title="Customer Support Triage RL") as demo:
    gr.Markdown("# 🎯 Customer Support Ticket Triage RL Environment")
    gr.Markdown("Train AI agents to classify and route support tickets!")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 🎮 Reset Episode")
            difficulty = gr.Dropdown(
                choices=["easy", "medium", "hard"],
                value="easy",
                label="Difficulty"
            )
            seed = gr.Number(label="Seed (optional)", value=0, precision=0)
            reset_btn = gr.Button("Reset Episode", variant="primary")
        
        with gr.Column():
            gr.Markdown("### 📊 Current Ticket")
            ticket_display = gr.Markdown(value="Click 'Reset Episode' to start")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 🎯 Your Action")
            route = gr.Dropdown(
                choices=["billing", "technical", "feature", "feedback", "spam"],
                label="Route Category"
            )
            urgency = gr.Dropdown(
                choices=["low", "medium", "high", "critical"],
                label="Urgency Level"
            )
            resolution_diff = gr.Dropdown(
                choices=["easy", "medium", "hard"],
                label="Resolution Difficulty"
            )
            priority = gr.Slider(0, 100, value=50, label="Priority Score")
            submit_btn = gr.Button("Submit Action", variant="primary")
        
        with gr.Column():
            gr.Markdown("### 📈 Metrics")
            progress = gr.Textbox(label="Progress", interactive=False)
            correctness = gr.Textbox(label="Correctness Score", interactive=False)
            efficiency = gr.Textbox(label="Efficiency Score", interactive=False)
            reward = gr.Textbox(label="Reward", interactive=False)
            done_status = gr.Textbox(label="Status", interactive=False)
    
    # Event handlers
    reset_btn.click(
        fn=reset_episode,
        inputs=[difficulty, seed],
        outputs=[ticket_display, progress, correctness, efficiency, reward]
    )
    
    submit_btn.click(
        fn=step_action,
        inputs=[route, urgency, resolution_diff, priority],
        outputs=[ticket_display, progress, correctness, efficiency, reward, done_status]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)