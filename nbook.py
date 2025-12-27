import marimo

__generated_with = "0.18.4"
app = marimo.App()


@app.cell
def _():
    from scripts.parser_procedure import Parser
    #from scripts.parser_entities import Tree, Element
    return (Parser,)


@app.cell
def _(Parser):
    input_string = 'raun'
    parser = Parser(level=1)
    res = parser.parse(input_string)
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
