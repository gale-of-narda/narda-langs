import sys
import typer
import logging

from rich.console import Console
from rich.logging import RichHandler

from scripts.parser_procedure import Processor


class ExitException(Exception):
    pass


def web_input(prompt_text):
    console.print(prompt_text, end="")

    result = []
    while True:
        char = sys.stdin.read(1)

        if not char:
            continue

        if char in ("\n", "\r"):
            sys.stdout.write("\r\n")
            sys.stdout.flush()
            break

        elif char in ("\b", "\x7f"):
            if result:
                result.pop()
                sys.stdout.write("\b \b")
                sys.stdout.flush()

        elif char == "\x1b":
            while True:
                next_char = sys.stdin.read(1)
                if next_char.isalpha() or next_char == "~":
                    break
            continue

        elif char.isprintable():
            result.append(char)
            sys.stdout.write(char)
            sys.stdout.flush()

    return "".join(result)


cli = typer.Typer(add_completion=False)
console = Console(force_terminal=True)
processor: Processor = None

logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, markup=True, show_path=False)],
)


@cli.command()
def process(text: str, verbose: bool = typer.Option(False, "--verbose", "-v")):
    res = processor.process(text, verbose=verbose)
    success = "[bold green]Successfully parsed[/bold green]"
    fail = "[bold red]Parsing failed for[/bold red]"
    res_string = success if res else fail
    console.print(f"{res_string} '{text}'")
    return


@cli.command()
def describe(verbose: bool = typer.Option(False, "--verbose", "-v")):
    processor.interpreter.describe(verbose=verbose)
    return


@cli.command()
def gloss():
    gloss_text = processor.interpreter.gloss()
    console.print(gloss_text)
    return


@cli.command()
def exit():
    raise ExitException


def main(path: str = ""):
    global processor
    try:
        processor = Processor(path=path)
    except Exception as e:
        console.print(f"[bold red]Error on load:[/]\n{e}")
        return

    console.rule("[bold green]Ready")

    while True:
        try:
            command = web_input("\n[bold blue]>[/] ")
            if not command.strip():
                continue
            cli(command.split(), standalone_mode=False)
        except ExitException:
            break
        except Exception as e:
            console.print(f"[red]Unexpected error:[/red] {e}")


if __name__ == "__main__":
    typer.run(main)
