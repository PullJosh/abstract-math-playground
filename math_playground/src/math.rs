use std::any::Any;
use std::collections::HashMap;
use std::fmt;

/*
    A NodeType describes a *type* of node. Each Node, then, references a particular node type
    and contains some data.
*/

trait NodeType2 {
    fn name(&self) -> &'static str;
}

struct AddType {}

// Nodes really represent the AST of a program. This includes math nodes,
// but also nodes to define variables, abstractions, etc.
impl NodeType2 for AddType {
    fn name(&self) -> &'static str {
        "Add"
    }
}

struct Node2 {
    node_type: Box<dyn NodeType2>,
    data: HashMap<String, Vec<u8>>, // Map from slot id (defined in node_type) to data
}

enum PrimitiveNodeType {
    U32,
    HeapPointer(String), // Pointer to a heap-allocated object
}

struct NodeType {
    layout: Vec<PrimitiveNodeType>,
}

pub trait Node: fmt::Display + fmt::Debug {
    fn eval_f64(&self) -> f64;
    fn matches(&self, other: &Box<dyn Node>) -> bool;
    fn as_any(&self) -> &dyn Any;
}

pub trait CommutativeNode: Node {}
pub trait AssociativeNode: Node {
    fn flatten(&self) -> Vec<Box<dyn Node>>;
}

#[derive(Debug)]
pub struct TemplatePlaceholder {}
impl TemplatePlaceholder {
    pub fn new() -> Self {
        TemplatePlaceholder {}
    }
}
impl Node for TemplatePlaceholder {
    fn eval_f64(&self) -> f64 {
        panic!("TemplatePlaceholder should not be evaluated");
    }
    fn matches(&self, _other: &Box<dyn Node>) -> bool {
        // The purpose of a TemplatePlaceholder is to match any other node
        true
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}
impl fmt::Display for TemplatePlaceholder {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "_")
    }
}

#[derive(Debug)]
pub struct Float {
    value: f64,
}
impl Float {
    pub fn new(value: f64) -> Self {
        Float { value }
    }
}
impl Node for Float {
    fn eval_f64(&self) -> f64 {
        self.value
    }
    fn matches(&self, other: &Box<dyn Node>) -> bool {
        if let Some(other) = other.as_any().downcast_ref::<Float>() {
            self.value == other.value
        } else {
            false
        }
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}
impl fmt::Display for Float {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.value)
    }
}

#[derive(Debug)]
pub struct Identifier {
    name: String,
}
impl Identifier {
    pub fn new(name: &str) -> Self {
        Identifier {
            name: name.to_string(),
        }
    }
}
impl Node for Identifier {
    fn eval_f64(&self) -> f64 {
        panic!("Identifier should not be evaluated");
    }
    fn matches(&self, other: &Box<dyn Node>) -> bool {
        if let Some(other) = other.as_any().downcast_ref::<Identifier>() {
            self.name == other.name
        } else {
            false
        }
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}
impl fmt::Display for Identifier {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.name)
    }
}

#[derive(Debug)]
pub struct Add {
    terms: Vec<Box<dyn Node>>,
}
impl Add {
    pub fn new(terms: Vec<Box<dyn Node>>) -> Self {
        Add { terms }
    }
}
impl Node for Add {
    fn eval_f64(&self) -> f64 {
        self.terms.iter().map(|term| term.eval_f64()).sum()
    }
    fn matches(&self, other: &Box<dyn Node>) -> bool {
        if let Some(other) = other.as_any().downcast_ref::<Add>() {
            if self.terms.len() != other.terms.len() {
                return false;
            }
            for (a, b) in self.terms.iter().zip(other.terms.iter()) {
                if !a.matches(b) {
                    return false;
                }
            }
            true
        } else {
            false
        }
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}
impl fmt::Display for Add {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "({})",
            self.terms
                .iter()
                .map(|term| format!("{}", term))
                .collect::<Vec<String>>()
                .join(" + ")
        )
    }
}

#[derive(Debug)]
pub struct Multiply {
    factors: Vec<Box<dyn Node>>,
}
impl Multiply {
    pub fn new(factors: Vec<Box<dyn Node>>) -> Self {
        Multiply { factors }
    }
}
impl Node for Multiply {
    fn eval_f64(&self) -> f64 {
        self.factors
            .iter()
            .map(|factor| factor.eval_f64())
            .product()
    }
    fn matches(&self, other: &Box<dyn Node>) -> bool {
        if let Some(other) = other.as_any().downcast_ref::<Multiply>() {
            if self.factors.len() != other.factors.len() {
                return false;
            }
            for (a, b) in self.factors.iter().zip(other.factors.iter()) {
                if !a.matches(b) {
                    return false;
                }
            }
            true
        } else {
            false
        }
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}
impl fmt::Display for Multiply {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "({})",
            self.factors
                .iter()
                .map(|factor| format!("{}", factor))
                .collect::<Vec<String>>()
                .join(" * ")
        )
    }
}
