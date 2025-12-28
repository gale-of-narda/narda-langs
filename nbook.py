import marimo

__generated_with = "0.18.4"
app = marimo.App()


@app.cell
def _():
    from scripts.parser_procedure import Parser
    from scripts.parser_entities import Tree, Element
    return (Parser,)


@app.cell
def _(Parser):
    input_string = "u u ı u o o e o"
    parser = Parser(level=0)
    parser.parse(input_string)
    print([e.stance for e in parser.buffer.mapping.elems])
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
