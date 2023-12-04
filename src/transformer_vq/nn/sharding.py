import jax


def get_namedsharding(device_mesh, axis_names):
    return jax.sharding.NamedSharding(
        device_mesh, jax.sharding.PartitionSpec(*axis_names)
    )


def sharding_constraint(x, device_mesh, axis_names):
    ns = get_namedsharding(device_mesh, axis_names)
    return jax.lax.with_sharding_constraint(x, ns)
