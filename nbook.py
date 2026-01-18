import marimo

__generated_with = "0.19.4"
app = marimo.App()


@app.cell
def _():
    from scripts.parser_procedure import Parser
    from scripts.parser_entities import Tree, Element
    return (Parser,)


@app.cell
def _(Parser):
    input_string = "ara"
    parser = Parser(level=0)
    parser.parse(input_string)
    print([e.stance for e in parser.buffer.mapping.elems])
    #for rank in parser.masker.masks[0]:
    #    for dich in rank:
    #        print(dich, dich.masks, dich.rev)
            #for mask in dich.masks:
            #    print(mask, mask.lemb)
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


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
