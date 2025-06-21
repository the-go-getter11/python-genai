import argparse
import asyncio
import os

from dotenv import load_dotenv
from rich.console import Console
from google import genai

console = Console()

async def pipeline_command(args: argparse.Namespace) -> None:
    console.print(f"[bold blue]Running pipeline:[/bold blue] {args.name}")
    # Placeholder for actual pipeline logic
    await asyncio.sleep(0)
    console.print("[green]Pipeline finished[/green]")

async def chat_command(args: argparse.Namespace) -> None:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        console.print("[red]GEMINI_API_KEY not set[/red]")
        return
    client = genai.Client(api_key=api_key)
    chat = client.aio.chats.create(model=args.model)
    response = await chat.send_message(args.message)
    console.print(f"[bold green]Response:[/bold green] {response.text}")

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GenAI command line interface")
    subparsers = parser.add_subparsers(dest="command", required=True)

    pipeline_parser = subparsers.add_parser("pipeline", help="Run a pipeline")
    pipeline_parser.add_argument("name", help="Name of the pipeline")
    pipeline_parser.set_defaults(func=pipeline_command)

    chat_parser = subparsers.add_parser("chat", help="Start a chat")
    chat_parser.add_argument("message", help="Prompt message")
    chat_parser.add_argument("--model", "-m", default="gemini-1.5-flash", help="Model name")
    chat_parser.set_defaults(func=chat_command)

    return parser

async def main() -> None:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args()
    await args.func(args)

if __name__ == "__main__":
    asyncio.run(main())
