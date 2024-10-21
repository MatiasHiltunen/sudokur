use ndarray::Array2;
use once_cell::sync::Lazy;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::env;
use std::fs::File;
use std::io::{self, BufRead, Write};
use std::path::Path;
use std::time::Instant;
use itertools::Itertools;

const N: usize = 9;
const MAX_ITERATIONS: usize = 20000;

// Set DEBUG to true to enable detailed logging
const DEBUG: bool = false;

// Bitmask constants for digits 1 through 9
const DIGIT_MASK: [u16; 10] = [
    0x000, // 0 (unused)
    0x001, // 1
    0x002, // 2
    0x004, // 3
    0x008, // 4
    0x010, // 5
    0x020, // 6
    0x040, // 7
    0x080, // 8
    0x100, // 9
];

type Matrix = Array2<u16>;

// Precompute peers for each cell
static PEERS: Lazy<Vec<Vec<(usize, usize)>>> = Lazy::new(|| {
    let mut peers = Vec::with_capacity(N * N);
    for i in 0..N {
        for j in 0..N {
            let mut peer_set = HashSet::with_capacity(20);

            // Add all cells in the same row, column, and box
            for k in 0..N {
                if k != j {
                    peer_set.insert((i, k));
                }
                if k != i {
                    peer_set.insert((k, j));
                }
            }

            let box_row_start = (i / 3) * 3;
            let box_col_start = (j / 3) * 3;
            for r in box_row_start..box_row_start + 3 {
                for c in box_col_start..box_col_start + 3 {
                    if r != i || c != j {
                        peer_set.insert((r, c));
                    }
                }
            }

            peers.push(peer_set.into_iter().collect());
        }
    }
    peers
});

/// Initialize the Sudoku matrix with the given puzzle.
fn init_sudoku_matrix(puzzle: [[u8; 9]; 9]) -> Matrix {
    let mut matrix = Array2::from_elem((N, N), 0x1FF);
    for i in 0..N {
        for j in 0..N {
            if let 1..=9 = puzzle[i][j] {
                matrix[[i, j]] = DIGIT_MASK[puzzle[i][j] as usize];
            }
        }
    }
    matrix
}

/// Validate the Sudoku puzzle to ensure no duplicates in any row, column, or 3x3 box.
fn validate_puzzle(puzzle: [[u8; 9]; 9]) -> bool {
    for i in 0..N {
        let mut row = HashSet::new();
        let mut col = HashSet::new();
        for j in 0..N {
            if puzzle[i][j] != 0 && !row.insert(puzzle[i][j]) {
                return false;
            }
            if puzzle[j][i] != 0 && !col.insert(puzzle[j][i]) {
                return false;
            }
        }
    }

    for box_row in (0..N).step_by(3) {
        for box_col in (0..N).step_by(3) {
            let mut box_set = HashSet::new();
            for i in 0..3 {
                for j in 0..3 {
                    let val = puzzle[box_row + i][box_col + j];
                    if val != 0 && !box_set.insert(val) {
                        return false;
                    }
                }
            }
        }
    }
    true
}

#[derive(Debug)]
enum PropagationResult {
    Success,
    Conflict,
}

/// Propagate constraints by eliminating candidates based on solved cells.
fn propagate_constraints(matrix: &mut Matrix) -> PropagationResult {
    let mut iterations = 0;

    loop {
        if iterations >= MAX_ITERATIONS {
            if DEBUG {
                println!("Reached maximum iterations in constraint propagation.");
            }
            break;
        }
        iterations += 1;

        let mut changed = false;

        // Step 1: Naked Singles
        let solved_cells: Vec<(usize, usize, u16)> = matrix
            .indexed_iter()
            .filter_map(|((i, j), &cell)| {
                if cell.count_ones() == 1 {
                    Some((i, j, cell))
                } else {
                    None
                }
            })
            .collect();

        for (i, j, cell) in solved_cells {
            for &(r, c) in &PEERS[i * N + j] {
                if matrix[[r, c]] & cell != 0 {
                    matrix[[r, c]] &= !cell;
                    changed = true;
                    if matrix[[r, c]] == 0 {
                        return PropagationResult::Conflict;
                    }
                }
            }
        }

        // Step 2: Hidden Singles
        for digit in 1..=9 {
            let mask = DIGIT_MASK[digit as usize];
            // Rows, Columns, and Boxes
            for unit in get_all_units() {
                let positions: Vec<(usize, usize)> = unit
                    .iter()
                    .filter(|&&(i, j)| matrix[[i, j]] & mask != 0)
                    .cloned()
                    .collect();
                if positions.len() == 1 {
                    let (i, j) = positions[0];
                    if matrix[[i, j]] != mask {
                        matrix[[i, j]] = mask;
                        changed = true;
                    }
                }
            }
        }

        // Step 3: Naked Pairs/Triples
        for unit in get_all_units() {
            let mut candidate_cells: HashMap<u16, Vec<(usize, usize)>> = HashMap::new();
            for &(i, j) in &unit {
                let cell = matrix[[i, j]];
                if (2..=3).contains(&cell.count_ones()) {
                    candidate_cells.entry(cell).or_default().push((i, j));
                }
            }
            for (&candidates, cells) in &candidate_cells {
                if cells.len() == candidates.count_ones() as usize {
                    for &(i, j) in &unit {
                        if !cells.contains(&(i, j)) {
                            let before = matrix[[i, j]];
                            matrix[[i, j]] &= !candidates;
                            if matrix[[i, j]] != before {
                                changed = true;
                                if matrix[[i, j]] == 0 {
                                    return PropagationResult::Conflict;
                                }
                            }
                        }
                    }
                }
            }
        }

        // Step 4: Hidden Pairs/Triples
        for unit in get_all_units() {
            let mut digit_positions: HashMap<u16, Vec<(usize, usize)>> = HashMap::new();
            for digit in 1..=9 {
                let mask = DIGIT_MASK[digit as usize];
                let positions: Vec<_> = unit
                    .iter()
                    .filter(|&&(i, j)| matrix[[i, j]] & mask != 0)
                    .cloned()
                    .collect();
                digit_positions.insert(digit, positions);
            }

            for count in 2..=3 {
                let digits: Vec<u16> = digit_positions
                    .iter()
                    .filter(|&(_, positions)| positions.len() == count)
                    .map(|(&digit, _)| digit)
                    .collect();

                for combination in digits.iter().combinations(count) {
                    let combined_positions: HashSet<_> = combination
                        .iter()
                        .flat_map(|&&digit| digit_positions[&digit].clone())
                        .collect();
                    if combined_positions.len() == count {
                        let mask = combination
                            .iter()
                            .fold(0u16, |acc, &&d| acc | DIGIT_MASK[d as usize]);
                        for &(i, j) in &combined_positions {
                            let before = matrix[[i, j]];
                            matrix[[i, j]] &= mask;
                            if matrix[[i, j]] != before {
                                changed = true;
                                if matrix[[i, j]] == 0 {
                                    return PropagationResult::Conflict;
                                }
                            }
                        }
                    }
                }
            }
        }

        // Step 5: Pointing Pairs/Triples
        for box_row in (0..N).step_by(3) {
            for box_col in (0..N).step_by(3) {
                let box_cells: Vec<(usize, usize)> = (box_row..box_row + 3)
                    .flat_map(|i| (box_col..box_col + 3).map(move |j| (i, j)))
                    .collect();

                for digit in 1..=9 {
                    let mask = DIGIT_MASK[digit as usize];
                    let positions: Vec<_> = box_cells
                        .iter()
                        .filter(|&&(i, j)| matrix[[i, j]] & mask != 0)
                        .cloned()
                        .collect();

                    if positions.len() >= 2 {
                        let same_row = positions.iter().all(|&(i, _)| i == positions[0].0);
                        let same_col = positions.iter().all(|&(_, j)| j == positions[0].1);

                        if same_row {
                            let row = positions[0].0;
                            for j in 0..N {
                                if (j < box_col || j >= box_col + 3)
                                    && matrix[[row, j]] & mask != 0
                                {
                                    matrix[[row, j]] &= !mask;
                                    changed = true;
                                    if matrix[[row, j]] == 0 {
                                        return PropagationResult::Conflict;
                                    }
                                }
                            }
                        } else if same_col {
                            let col = positions[0].1;
                            for i in 0..N {
                                if (i < box_row || i >= box_row + 3)
                                    && matrix[[i, col]] & mask != 0
                                {
                                    matrix[[i, col]] &= !mask;
                                    changed = true;
                                    if matrix[[i, col]] == 0 {
                                        return PropagationResult::Conflict;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        if !changed {
            break;
        }
    }

    PropagationResult::Success
}

/// Get all units (rows, columns, and boxes) of the Sudoku grid.
fn get_all_units() -> Vec<Vec<(usize, usize)>> {
    let mut units = Vec::new();

    // Rows
    for i in 0..N {
        units.push((0..N).map(|j| (i, j)).collect());
    }

    // Columns
    for j in 0..N {
        units.push((0..N).map(|i| (i, j)).collect());
    }

    // Boxes
    for box_row in (0..N).step_by(3) {
        for box_col in (0..N).step_by(3) {
            units.push(
                (box_row..box_row + 3)
                    .flat_map(|i| (box_col..box_col + 3).map(move |j| (i, j)))
                    .collect(),
            );
        }
    }

    units
}

#[derive(Debug)]
enum SolveResult {
    Solved,
    Unsolvable,
}

/// Solve the Sudoku puzzle using backtracking with constraint propagation.
fn solve_sudoku(matrix: &mut Matrix) -> SolveResult {
    match propagate_constraints(matrix) {
        PropagationResult::Conflict => return SolveResult::Unsolvable,
        PropagationResult::Success => {}
    }

    if matrix.iter().all(|&cell| cell.count_ones() == 1) {
        return SolveResult::Solved;
    }

    if let Some((i, j, candidates)) = find_least_candidates(matrix) {
        for &digit in &candidates {
            let mut matrix_copy = matrix.clone();
            matrix_copy[[i, j]] = DIGIT_MASK[digit as usize];
            if let SolveResult::Solved = solve_sudoku(&mut matrix_copy) {
                *matrix = matrix_copy;
                return SolveResult::Solved;
            }
        }
    }

    SolveResult::Unsolvable
}

/// Find the cell with the fewest candidates (minimum remaining value heuristic).
fn find_least_candidates(matrix: &Matrix) -> Option<(usize, usize, Vec<u16>)> {
    matrix
        .indexed_iter()
        .filter(|&(_, &cell)| cell.count_ones() > 1)
        .min_by_key(|&(_, &cell)| cell.count_ones())
        .map(|((i, j), &cell)| {
            let candidates = (1..=9)
                .filter(|&d| cell & DIGIT_MASK[d as usize] != 0)
                .collect();
            (i, j, candidates)
        })
}

/// Serialize the solved puzzle into a string with digits only.
fn serialize_solved_puzzle(matrix: &Matrix) -> String {
    (0..N)
        .flat_map(|i| {
            (0..N).map(move |j| {
                if matrix[[i, j]].count_ones() == 1 {
                    char::from_digit((matrix[[i, j]].trailing_zeros() + 1) as u32, 10).unwrap()
                } else {
                    '0'
                }
            })
        })
        .collect()
}

/// Serialize the unsolved puzzle into a string with digits and zeros.
fn serialize_unsolved_puzzle(puzzle: [[u8; 9]; 9]) -> String {
    puzzle
        .iter()
        .flatten()
        .map(|&cell| {
            if cell == 0 {
                '0'
            } else {
                char::from_digit(cell as u32, 10).unwrap()
            }
        })
        .collect()
}

/// Read Sudoku puzzles from a file.
fn read_puzzles_from_file<P: AsRef<Path>>(filename: P) -> io::Result<Vec<[[u8; 9]; 9]>> {
    let file = File::open(filename)?;
    let reader = io::BufReader::new(file);
    let mut puzzles = Vec::new();

    for (line_num, line_result) in reader.lines().enumerate() {
        let line = line_result?;
        let trimmed = line.trim();

        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        if trimmed.len() != 81 {
            eprintln!(
                "Warning: Line {} is invalid (expected 81 characters, got {}). Skipping.",
                line_num + 1,
                trimmed.len()
            );
            continue;
        }

        if let Some(puzzle) = parse_line(trimmed) {
            puzzles.push(puzzle);
        }
    }

    Ok(puzzles)
}

/// Parse a line of 81 characters into a 9x9 Sudoku puzzle.
fn parse_line(trimmed: &str) -> Option<[[u8; 9]; 9]> {
    let mut puzzle = [[0u8; 9]; 9];

    for (idx, ch) in trimmed.chars().enumerate() {
        let row = idx / 9;
        let col = idx % 9;

        puzzle[row][col] = match ch {
            '.' | '0' => 0,
            '1'..='9' => ch.to_digit(10).unwrap() as u8,
            _ => {
                eprintln!("Warning: Invalid character '{}'. Skipping line.", ch);
                return None;
            }
        };
    }

    Some(puzzle)
}

/// Deserialize the matrix into a 9x9 grid.
fn deserialize_matrix(matrix: &Matrix) -> [[u8; 9]; 9] {
    let mut puzzle = [[0u8; 9]; 9];
    for ((i, j), &cell) in matrix.indexed_iter() {
        if cell.count_ones() == 1 {
            puzzle[i][j] = (cell.trailing_zeros() + 1) as u8;
        }
    }
    puzzle
}

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() != 2 {
        eprintln!(
            "Usage: {} <puzzles_file>",
            args.get(0).map(|s| s.as_str()).unwrap_or("sudoku_solver")
        );
        std::process::exit(1);
    }

    let filename = &args[1];

    let puzzles = match read_puzzles_from_file(filename) {
        Ok(puzzles) => puzzles,
        Err(e) => {
            eprintln!("Error reading file '{}': {}", filename, e);
            std::process::exit(1);
        }
    };

    if puzzles.is_empty() {
        println!("No valid puzzles found in the file '{}'.", filename);
        return;
    }

    println!("Found {} puzzle(s)\n", puzzles.len());

    let total_start = Instant::now();

    let solved = puzzles
        .par_iter()
        .map(|&puzzle| {
            let mut matrix = init_sudoku_matrix(puzzle);

            if !validate_puzzle(puzzle) {
                return format!(
                    "{},Invalid initial puzzle with duplicates detected.",
                    serialize_unsolved_puzzle(puzzle)
                );
            }

            let unsolved = serialize_unsolved_puzzle(puzzle);

            match solve_sudoku(&mut matrix) {
                SolveResult::Solved => {
                    let solved_puzzle = deserialize_matrix(&matrix);
                    if validate_puzzle(solved_puzzle) {
                        let solved_serialized = serialize_solved_puzzle(&matrix);
                        format!("{},{}", unsolved, solved_serialized)
                    } else {
                        format!(
                            "{},Invalid solution with duplicates detected.",
                            unsolved
                        )
                    }
                }
                SolveResult::Unsolvable => format!("{},No solution exists.", unsolved),
            }
        })
        .collect::<Vec<String>>();

    let total_duration = total_start.elapsed();

    let output_filename = "result.txt";
    match write_output_file(output_filename, &solved, puzzles.len() as u32) {
        Ok(_) => {
            println!(
                "All puzzles have been processed. Solutions are saved in '{}'.",
                output_filename
            );
        }
        Err(e) => {
            eprintln!("Error writing to file '{}': {}", output_filename, e);
            std::process::exit(1);
        }
    }

    println!(
        "\nTime for solving {} puzzle(s): {:.6} seconds\n",
        puzzles.len(),
        total_duration.as_secs_f64()
    );
}

/// Write the solved puzzles to an output file.
fn write_output_file<P: AsRef<Path>>(
    filename: P,
    solved: &[String],
    count: u32,
) -> io::Result<()> {
    let mut file = File::create(filename)?;

    writeln!(&mut file, "{}", count)?;
    for solved_entry in solved.iter() {
        writeln!(&mut file, "{}", solved_entry)?;
    }

    Ok(())
}
