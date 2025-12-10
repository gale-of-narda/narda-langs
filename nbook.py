import marimo

__generated_with = "0.18.4"
app = marimo.App()


@app.cell
def _():
    from scripts.parser_procedure import Parser
    return (Parser,)


@app.cell
def _(Parser):
    input_string = 'rō'
    parser = Parser(level=1)
    res = parser.parse(input_string)
    print(res)
    return


@app.cell
def _():
    return


@app.cell
def _():
    #for rank in parser.masker.masks:
    #    for mask_pair in rank:
    #        for mask in mask_pair:
    #            print(mask, mask.rev)
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
