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
    #rémı
    #raéan
    #hréag
    #mı́ıfasol
    input_string = 'doparı́ıf'
    parser = Parser(level=1)
    res = parser.parse(input_string)
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
