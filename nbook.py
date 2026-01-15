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
    input_string = "ra"
    parser = Parser(level=0)
    parser.parse(input_string)
    print([e.stance for e in parser.buffer.mapping.elems])
    #for rank in parser.masker.masks[0]:
    #    for mask_pair in rank:
    #        for mask in mask_pair:
    #            print(mask, mask.key, mask.rep)
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
