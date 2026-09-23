//! The reified transition.

mod buf;
mod entry;
mod lifecycle;
mod stack;

pub use buf::{DeltaBuf, MoveKey, OutOfPlane, Resolve, SlotRef};
pub use entry::{EndImage, Entry, MoveHeader};
pub use lifecycle::{Applied, Delta, LevelIter, Receipt, Recording, Transition};
pub use stack::{DeltaStack, Workspace};
