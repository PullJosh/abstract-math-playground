use math_playground_lib::run;

mod math;
use math::*;
use std::collections::HashMap;

fn main() {
    // let node = Add::new(vec![
    //     Box::new(Add::new(vec![
    //         Box::new(Identifier::new("x")),
    //         Box::new(Identifier::new("y")),
    //     ])),
    //     Box::new(Float::new(3.0)),
    // ]);
    // log::info!("{:?}", node);

    pollster::block_on(run());

    // Each module contains a bunch of stuff. There is no
    // distinction between state and code–all of it is just stuff.
    // let std_library = Module {
    //     stuff: HashMap::from([(
    //         uuid!("cebc2d43-5bea-41e9-a996-07a4e178584a"),
    //         Value {
    //             data_type: PrimitiveDataType::U32,
    //             // Raw binary data for number 42
    //             data: &[0x2a], // 2 * 16 + 10 = 42
    //         },
    //     )]),
    //     public: vec![uuid!("cebc2d43-5bea-41e9-a996-07a4e178584a")],
    // };

    // let add_node_type = NodeType {
    //     name: "Add",
    //     slots: HashMap::from([(
    //         "terms",
    //         SlotType {
    //             name: "terms",
    //             node_type: "Node",
    //             multiple: true,
    //             commutative: true, // Means that elmenets of this slot can safely be reordered
    //         },
    //     )]),
    // };

    // let polynomial_term_type = NodeType {
    //     name: "PolynomialTerm",
    //     slots: HashMap::from([
    //         (
    //             "coefficient",
    //             SlotType {
    //                 name: "coefficient",
    //                 node_type: "Node",
    //             },
    //         ),
    //         (
    //             "variable",
    //             SlotType {
    //                 name: "variable",
    //                 node_type: "Identifier",
    //             },
    //         ),
    //         (
    //             "exponent",
    //             SlotType {
    //                 name: "exponent",
    //                 node_type: "Node",
    //             },
    //         ),
    //     ]),
    // };

    // let polynomial_type = NodeType {
    //     name: "Polynomial",
    //     slots: HashMap::from([(
    //         "terms",
    //         SlotType {
    //             name: "terms",
    //             node_type: "PolynomialTerm",
    //             multiple: true,
    //         },
    //     )]),
    // };
}
