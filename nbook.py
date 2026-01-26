import marimo

__generated_with = "0.19.6"
app = marimo.App()


@app.cell
def _():
    from scripts.parser_procedure import Parser
    #from scripts.parser_entities import Tree, Element
    return (Parser,)


@app.cell
def _(Parser):
    parser = Parser(level=0)
    return (parser,)


@app.cell
def _(parser):
    input_string = "máınuras"
    parser.process(input_string)
    print(parser.interpreter.tree)
    print([e.stance for e in parser.mapping.elems])
    #for rank in parser.masker.masks[0]:
    #    for dich in rank:
    #        print(dich, dich.preterminal, dich.terminal, dich.d)
    #        for mask in dich.masks:
    #            print(mask, mask.active)
    return


@app.cell
def _(parser):
    #parser.interpreter.tree.all_nodes
    parser.interpreter.describe(verbose=False)
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
