use ndarray::Array2;
use once_cell::sync::Lazy;
use rayon::prelude::*;
use std::collections::HashSet;
use std::fs::File;
use std::io::{self, BufRead, Write};
use std::path::Path;
use std::time::Instant;

const N: usize = 9;
const MAX_ITERATIONS: usize = 20000;

// Set DEBUG to true to enable detailed logging
const DEBUG: bool = true;

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
            let mut peer_set = HashSet::with_capacity(20); // Each cell has 20 unique peers

            // Add all cells in the same row
            for col in 0..N {
                if col != j {
                    peer_set.insert((i, col));
                }
            }

            // Add all cells in the same column
            for row in 0..N {
                if row != i {
                    peer_set.insert((row, j));
                }
            }

            // Add all cells in the same 3x3 box
            let box_row_start = (i / 3) * 3;
            let box_col_start = (j / 3) * 3;
            for r in box_row_start..box_row_start + 3 {
                for c in box_col_start..box_col_start + 3 {
                    if r != i || c != j {
                        peer_set.insert((r, c));
                    }
                }
            }

            let peer_vec: Vec<(usize, usize)> = peer_set.into_iter().collect();
            // Verify that each cell has exactly 20 unique peers
            assert_eq!(
                peer_vec.len(),
                20,
                "Cell ({}, {}) does not have 20 peers.",
                i,
                j
            );

            peers.push(peer_vec);
        }
    }

    // Ensure no cell includes itself as a peer
    for i in 0..N {
        for j in 0..N {
            assert!(
                !peers[i * N + j].contains(&(i, j)),
                "Cell ({}, {}) incorrectly includes itself as a peer.",
                i,
                j
            );
        }
    }

    peers
});

/// Initialize the Sudoku matrix with the given puzzle.
/// Known digits are set to their corresponding bitmask, and unknown cells are set to 0x1FF (all candidates).
fn init_sudoku_matrix(puzzle: [[u8; 9]; 9]) -> Matrix {
    let mut matrix = Array2::<u16>::from_elem((N, N), 0x1FF); // All candidates initially

    for i in 0..N {
        for j in 0..N {
            let val = puzzle[i][j];
            if val >= 1 && val <= 9 {
                matrix[[i, j]] = DIGIT_MASK[val as usize]; // Set known digit
                if DEBUG {
                    println!("Setting cell ({}, {}) to digit {}", i, j, val);
                }
            }
        }
    }
    matrix
}

/// Validate the Sudoku puzzle to ensure no duplicates in any row, column, or 3x3 box.
/// Returns `true` if the puzzle is valid, `false` otherwise.
fn validate_puzzle(puzzle: [[u8; 9]; 9]) -> bool {
    for i in 0..9 {
        let mut row = HashSet::new();
        let mut col = HashSet::new();
        let mut box_set = HashSet::new();
        for j in 0..9 {
            // Validate Row
            if puzzle[i][j] != 0 {
                if !row.insert(puzzle[i][j]) {
                    println!("Duplicate digit {} found in row {}.", puzzle[i][j], i + 1);
                    return false;
                }
            }

            // Validate Column
            if puzzle[j][i] != 0 {
                if !col.insert(puzzle[j][i]) {
                    println!(
                        "Duplicate digit {} found in column {}.",
                        puzzle[j][i],
                        i + 1
                    );
                    return false;
                }
            }

            // Validate 3x3 Box
            let box_row = 3 * (i / 3) + j / 3;
            let box_col = 3 * (i % 3) + j % 3;
            if puzzle[box_row][box_col] != 0 {
                if !box_set.insert(puzzle[box_row][box_col]) {
                    println!(
                        "Duplicate digit {} found in box starting at ({}, {}).",
                        puzzle[box_row][box_col],
                        (box_row / 3) * 3,
                        (box_col / 3) * 3
                    );
                    return false;
                }
            }
        }
    }
    true
}

/// Propagate constraints by eliminating candidates based on solved cells.
/// Returns `true` if successful, `false` if a conflict is detected.
fn propagate_constraints(matrix: &mut Matrix) -> bool {
    let start_time = Instant::now();
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

        // Step 1: Process Naked Singles
        let solved_cells: Vec<(usize, usize, u16)> = matrix
            .indexed_iter()
            .filter_map(|((i, j), &cell)| {
                if cell.count_ones() == 1 {
                    Some((i, j, (cell.trailing_zeros() as u16) + 1)) // 1-based digit
                } else {
                    None
                }
            })
            .collect();

        if DEBUG {
            println!("Iteration {}: Solved Cells: {:?}", iterations, solved_cells);
        }

        for (i, j, digit) in solved_cells {
            // Eliminate the digit from all peers
            for &(r, c) in &PEERS[i * N + j] {
                if matrix[[r, c]].count_ones() == 1 && matrix[[r, c]] == DIGIT_MASK[digit as usize] {
                    continue;
                }
                if (matrix[[r, c]] & DIGIT_MASK[digit as usize]) != 0 {
                    matrix[[r, c]] &= !DIGIT_MASK[digit as usize];
                    changed = true;
                    if matrix[[r, c]].count_ones() == 0 {
                        if DEBUG {
                            println!("Conflict detected at cell ({}, {})", r, c);
                        }
                        return false;
                    }
                }
            }
        }

        // Step 2: Process Hidden Singles
        for digit in 1..=9 {
            let mask = DIGIT_MASK[digit as usize];

            // Check rows
            for i in 0..N {
                let mut positions = vec![];
                for j in 0..N {
                    if (matrix[[i, j]] & mask) != 0 {
                        positions.push((i, j));
                    }
                }
                if positions.len() == 1 {
                    let (i, j) = positions[0];
                    if matrix[[i, j]].count_ones() != 1 {
                        matrix[[i, j]] = mask;
                        changed = true;
                        if DEBUG {
                            println!("Hidden single for digit {} in row {} at cell ({}, {})", digit, i + 1, i, j);
                        }
                    }
                }
            }

            // Check columns
            for j in 0..N {
                let mut positions = vec![];
                for i in 0..N {
                    if (matrix[[i, j]] & mask) != 0 {
                        positions.push((i, j));
                    }
                }
                if positions.len() == 1 {
                    let (i, j) = positions[0];
                    if matrix[[i, j]].count_ones() != 1 {
                        matrix[[i, j]] = mask;
                        changed = true;
                        if DEBUG {
                            println!("Hidden single for digit {} in column {} at cell ({}, {})", digit, j + 1, i, j);
                        }
                    }
                }
            }

            // Check boxes
            for box_row in (0..N).step_by(3) {
                for box_col in (0..N).step_by(3) {
                    let mut positions = vec![];
                    for i in box_row..box_row + 3 {
                        for j in box_col..box_col + 3 {
                            if (matrix[[i, j]] & mask) != 0 {
                                positions.push((i, j));
                            }
                        }
                    }
                    if positions.len() == 1 {
                        let (i, j) = positions[0];
                        if matrix[[i, j]].count_ones() != 1 {
                            matrix[[i, j]] = mask;
                            changed = true;
                            if DEBUG {
                                println!("Hidden single for digit {} in box starting at ({}, {}) at cell ({}, {})", digit, box_row + 1, box_col + 1, i, j);
                            }
                        }
                    }
                }
            }
        }

        // Termination condition
        if !changed {
            if DEBUG {
                println!("No changes made in iteration {}. Terminating constraint propagation.", iterations);
            }
            break;
        }
    }

    let elapsed = start_time.elapsed();
    if DEBUG {
        println!(
            "Time for constraint propagation: {:.6} seconds, Iterations: {}",
            elapsed.as_secs_f64(),
            iterations
        );
    }
    true
}

/// Find the cell with the fewest candidates (minimum remaining value heuristic).
/// Returns the cell's coordinates and its possible candidates.
fn find_least_candidates(matrix: &Matrix) -> Option<(usize, usize, Vec<u16>)> {
    let mut min_candidates = 10;
    let mut target = None;

    for ((i, j), &cell) in matrix.indexed_iter() {
        let count = cell.count_ones();
        if count > 1 && count < min_candidates {
            min_candidates = count;
            let candidates: Vec<u16> = (1..=9)
                .filter(|&d| (cell & DIGIT_MASK[d as usize]) != 0)
                .collect();
            target = Some((i, j, candidates));
            if count == 2 {
                break; // Optimal choice
            }
        }
    }
    target
}

/// Solve the Sudoku puzzle using backtracking with constraint propagation.
/// Returns `true` if a solution is found, `false` otherwise.
/// The matrix is modified in place with the solution if found.
fn solve_sudoku(matrix: &mut Matrix) -> bool {
    // First, propagate constraints
    if !propagate_constraints(matrix) {
        return false; // Conflict detected
    }

    // Check if the puzzle is solved
    if matrix.iter().all(|&cell| cell.count_ones() == 1) {
        return true;
    }

    // Choose the cell with the fewest candidates
    if let Some((i, j, candidates)) = find_least_candidates(matrix) {
        if DEBUG {
            println!(
                "Backtracking: Assigning to cell ({}, {}) with candidates {:?}",
                i, j, candidates
            );
        }

        for &digit in &candidates {
            if DEBUG {
                println!(
                    "Attempting to assign digit {} to cell ({}, {})",
                    digit, i, j
                );
            }

            // Save the current state of the cell
            let previous = matrix[[i, j]];

            // Assign the digit to the cell
            matrix[[i, j]] = DIGIT_MASK[digit as usize];

            // Recursively attempt to solve the updated puzzle
            if solve_sudoku(matrix) {
                return true;
            }

            if DEBUG {
                println!(
                    "Backtracking: Digit {} at cell ({}, {}) led to conflict, reverting.",
                    digit, i, j
                );
            }

            // Revert the assignment
            matrix[[i, j]] = previous;
        }
    }

    false
}

/// Print the Sudoku matrix in a readable format.
fn print_sudoku(matrix: &Matrix) {
    for i in 0..N {
        for j in 0..N {
            if matrix[[i, j]].count_ones() == 1 {
                let digit = (matrix[[i, j]].trailing_zeros() as u16) + 1;
                print!("{} ", digit);
            } else {
                print!(". ");
            }
            if (j + 1) % 3 == 0 && j < 8 {
                print!("| ");
            }
        }
        println!();
        if (i + 1) % 3 == 0 && i < 8 {
            println!("------+-------+------");
        }
    }
}

/// Verify that the digit masks are correctly defined.
fn verify_digit_masks() {
    for d in 1..=9 {
        assert_eq!(
            DIGIT_MASK[d],
            1 << (d - 1),
            "DIGIT_MASK for digit {} is incorrect.",
            d
        );
    }
    println!("All digit masks verified successfully.");
}

/// Parse a line of 81 characters into a 9x9 Sudoku puzzle.
/// Returns `Some(puzzle)` if parsing is successful, `None` otherwise.
fn parse_line(trimmed: &str) -> Option<[[u8; 9]; 9]> {
    // Parse the line into a 9x9 grid
    let mut puzzle = [[0u8; 9]; 9];
    let mut valid = true;

    for (idx, ch) in trimmed.chars().enumerate() {
        let row = idx / 9;
        let col = idx % 9;

        if ch == '.' || ch == '0' {
            puzzle[row][col] = 0; // Represent empty cells with 0
        } else if ch.is_digit(10) {
            let digit = ch.to_digit(10).unwrap() as u8;
            if digit >= 1 && digit <= 9 {
                puzzle[row][col] = digit;
            } else {
                eprintln!("Warning: Line contains invalid digit '{}'. Skipping.", ch);
                valid = false;
                break;
            }
        } else {
            eprintln!(
                "Warning: Line contains invalid character '{}'. Skipping.",
                ch
            );
            valid = false;
            break;
        }
    }

    if valid {
        Some(puzzle)
    } else {
        None
    }
}

/// Read Sudoku puzzles from a file.
/// Each puzzle should be a line with exactly 81 characters.
/// Returns a vector of puzzles.
fn read_puzzles_from_file<P: AsRef<Path>>(filename: P) -> io::Result<Vec<[[u8; 9]; 9]>> {
    let file = File::open(filename)?;
    let reader = io::BufReader::new(file);
    let mut puzzles = Vec::new();

    for (line_num, line_result) in reader.lines().enumerate() {
        let line = line_result?;
        let trimmed = line.trim();

        // Skip empty lines and comment lines
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

/// Serialize the unsolved puzzle into a string with digits and zeros.
fn serialize_unsolved_puzzle(puzzle: [[u8; 9]; 9]) -> String {
    let mut s = String::with_capacity(81);
    for row in &puzzle {
        for &cell in row {
            s.push(if cell == 0 {
                '0'
            } else {
                char::from_digit(cell as u32, 10).unwrap()
            });
        }
    }
    s
}

/// Serialize the solved puzzle into a string with digits only.
fn serialize_solved_puzzle(matrix: &Matrix) -> String {
    let mut s = String::with_capacity(81);
    for i in 0..N {
        for j in 0..N {
            let digit = if matrix[[i, j]].count_ones() == 1 {
                (matrix[[i, j]].trailing_zeros() as u8) + 1
            } else {
                0
            };
            s.push(char::from_digit(digit as u32, 10).unwrap());
        }
    }
    s
}

/// Write the solved puzzles to an output file.
/// The file starts with the count of puzzles, followed by each puzzle and its solution.
fn write_output_file<P: AsRef<Path>>(filename: P, solved: &[String], count: u32) -> io::Result<()> {
    let mut file = File::create(filename)?;

    // Write the count of puzzles
    writeln!(&mut file, "{}", count)?;

    // Write each puzzle and its solution or error message
    for solved_entry in solved.iter() {
        writeln!(&mut file, "{}", solved_entry)?;
    }

    // Ensure the file ends with a single newline
    Ok(())
}

/// Deserialize the matrix into a 9x9 grid.
fn deserialize_matrix(matrix: &Matrix) -> [[u8; 9]; 9] {
    let mut puzzle = [[0u8; 9]; 9];
    for i in 0..N {
        for j in 0..N {
            if matrix[[i, j]].count_ones() == 1 {
                puzzle[i][j] = (matrix[[i, j]].trailing_zeros() as u8) + 1;
            } else {
                puzzle[i][j] = 0; // Should not happen if solved correctly
            }
        }
    }
    puzzle
}

fn main() {
    // Verify that digit masks are correctly defined
    verify_digit_masks();

    /*
    Uncomment the following block to read puzzles from a file.
    Make sure to provide the filename as a command-line argument.

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
    */

    // Example puzzle for demonstration
    let puzzles = [parse_line(
        "000075400000000008080190000300001060000000034000068170204000603900000020530200000",
    )
    .unwrap()];

    println!("Found {} puzzle(s)\n", puzzles.len());

    let total_start = Instant::now();

    let solved = puzzles
        .par_iter()
        .map(|&puzzle| {
            // Initialize the matrix and solve
            let mut matrix = init_sudoku_matrix(puzzle);

            // Validate the initial puzzle
            if !validate_puzzle(puzzle) {
                return format!("{},Invalid initial puzzle with duplicates detected.", serialize_unsolved_puzzle(puzzle));
            }

            let unsolved = serialize_unsolved_puzzle(puzzle);

            // Solve the puzzle
            let is_solved = solve_sudoku(&mut matrix);

            if is_solved {
                // Deserialize the matrix to validate the solution
                let solved_puzzle = deserialize_matrix(&matrix);
                if validate_puzzle(solved_puzzle) {
                    let solved_serialized = serialize_solved_puzzle(&matrix);
                    format!("{},{}", unsolved, solved_serialized)
                } else {
                    format!("{},Invalid solution with duplicates detected.", unsolved)
                }
            } else {
                format!("{},No solution exists for the given Sudoku.", unsolved)
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

            // Optional: Verify the checksums (requires external tools or additional Rust crates)
            println!(
                "Please verify the MD5 and SHA256 checksums externally to ensure correctness."
            );
        }
        Err(e) => {
            eprintln!("Error writing to file '{}': {}", output_filename, e);
            std::process::exit(1);
        }
    }

    println!(
        "\nTime for solving Puzzle {}: {:.6} seconds\n",
        puzzles.len(),
        total_duration.as_secs_f64()
    );
}
