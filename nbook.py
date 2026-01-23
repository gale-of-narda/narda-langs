import marimo

__generated_with = "0.19.5"
app = marimo.App()


@app.cell
def _():
    from scripts.parser_procedure import Parser
    #from scripts.parser_entities import Tree, Element
    return (Parser,)


@app.cell
def _():
    import random

    alphabet="eanbt"
    length = 5
    count = 10

    def generate_random_strings(count, length, alphabet):
        return [''.join(random.choices(alphabet, k=length)) for _ in range(count)]

    result = generate_random_strings(count, length, alphabet)
    print(result)
    return


@app.cell
def _(Parser):
    input_string = "takunádo"
    parser = Parser(level=0)
    parser.parse(input_string)
    print([e.stance for e in parser.buffer.mapping.elems])
    #for rank in parser.masker.masks[0]:
    #    for dich in rank:
    #        print(dich, dich.preterminal, dich.terminal, dich.d)
    #        for mask in dich.masks:
    #            print(mask, mask.active)
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
