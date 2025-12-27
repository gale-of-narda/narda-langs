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
    input_string = '́ann'
    parser = Parser(level=1)
    parser.parse(input_string)
    stances = parser.buffer.mapping.stances
    print([[[pos for pos in comp] for comp in st] for st in stances])
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
