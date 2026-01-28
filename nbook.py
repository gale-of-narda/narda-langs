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
    input_string = "mán-àve"
    parser.process(input_string)
    #for rank in parser.masker.masks[0]:
    #    for dich in rank:
    #        print(dich, dich.preterminal, dich.terminal, dich.d)
    #        for mask in dich.masks:
    #            print(mask, mask.active)
    return


@app.cell
def _(parser):
    print([e.stance for e in parser.mapping.elems])
    parser.interpreter.draw_tree(all_nodes=True, features=True)
    return


@app.cell
def _(parser):
    parser.interpreter.describe()
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
