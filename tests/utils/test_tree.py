from transformer_vq.utils.tree import flattened_traversal


def test_flattened_traversal():
    def func(k, v):
        return f"{'.'.join(k)}_{v}"

    pytree1 = {"a": 1, "b": 2, "c": 3}
    actual = flattened_traversal(func)(pytree1)
    expected = {"a": "a_1", "b": "b_2", "c": "c_3"}
    assert actual == expected

    pytree2 = {"a": 1, "b": 2, "c": {"d": {"e": 3}}}
    actual = flattened_traversal(func)(pytree2)
    expected = {"a": "a_1", "b": "b_2", "c": {"d": {"e": "c.d.e_3"}}}
    assert actual == expected
