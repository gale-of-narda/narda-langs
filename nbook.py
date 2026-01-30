import marimo

__generated_with = "0.19.7"
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
    input_string = ["ránu<sán"]
    parser.process(input_string)
    #for rank in parser.masker.masks[0]:
    #    for dich in rank:
    #        print(dich)
    #        for mask in dich.masks:
    #            print(mask, mask.demb, mask.rev)
    return


@app.cell
def _(parser):
    print([e.stance for e in parser.mappings[0].elems])
    parser.draw_tree(parser.trees[-1], all_nodes=True, features=True)
    return


@app.cell
def _(parser):
    parser.interpreter.describe(parser.trees[-1])
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
