use ndarray::Array2;
use once_cell::sync::Lazy;
use std::collections::HashSet;
use std::env;
use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;
use std::time::Instant;
use rayon::prelude::*;


const N: usize = 9;
const MAX_ITERATIONS: usize = 1000; 


const DEBUG: bool = false; 

// Bitmask constants
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

            for col in 0..N {
                if col != j {
                    peer_set.insert((i, col));
                }
            }


            for row in 0..N {
                if row != i {
                    peer_set.insert((row, j));
                }
            }

        
            let box_row_start = (i / 3) * 3;
            let box_col_start = (j / 3) * 3;
            for r in box_row_start..box_row_start + 3 {
                for c in box_col_start..box_col_start + 3 {
                    if r != i || c != j {
                        // Correctly exclude the cell itself
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


fn init_sudoku_matrix(puzzle: [[u8; 9]; 9]) -> Matrix {
    // all candidates initially (bits 0-8 set)
    let mut matrix = Array2::<u16>::from_elem((N, N), 0x1FF); 

    for i in 0..N {
        for j in 0..N {
            let val = puzzle[i][j];
            if val >= 1 && val <= 9 {
                matrix[[i, j]] = DIGIT_MASK[val as usize]; // bitmask for the digit
                if DEBUG {
                    println!("Setting cell ({}, {}) to digit {}", i, j, val);
                }
            }
        }
    }
    matrix
}

fn validate_puzzle(puzzle: [[u8; 9]; 9]) -> bool {
    for i in 0..9 {
        let mut row = HashSet::new();
        let mut col = HashSet::new();
        let mut box_set = HashSet::new();
        for j in 0..9 {
   
            if puzzle[i][j] != 0 {
                if !row.insert(puzzle[i][j]) {
                    println!("Duplicate digit {} found in row {}.", puzzle[i][j], i + 1);
                    return false;
                }
            }
     
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

        // Find all cells with a single candidate
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

        if solved_cells.is_empty() {
            break;
        }

        if DEBUG {
            println!("Iteration {}: Solved Cells: {:?}", iterations, solved_cells);
        }

        // Flag to track if any changes were made in this iteration
        let mut changed = false;

        for (i, j, digit) in solved_cells {
            if DEBUG {
                println!(
                    "Eliminating digit {} from peers of cell ({}, {})",
                    digit, i, j
                );
            }

            // Eliminate the digit from all peers
            for &(r, c) in &PEERS[i * N + j] {
                // Skip elimination if the peer is already solved
                if matrix[[r, c]].count_ones() == 1 {
                    if DEBUG {
                        println!("Skipping elimination for solved cell ({}, {})", r, c);
                    }
                    continue;
                }

                if (matrix[[r, c]] & DIGIT_MASK[digit as usize]) != 0 {
                    if DEBUG {
                        println!("Eliminating digit {} from cell ({}, {})", digit, r, c);
                    }
                    matrix[[r, c]] &= !DIGIT_MASK[digit as usize];
                    changed = true; // Mark that a change has occurred

                    // If a cell is reduced to zero candidates, there's a conflict
                    if matrix[[r, c]].count_ones() == 0 {
                        if DEBUG {
                            println!("Conflict detected at cell ({}, {})", r, c);
                        }
                        return false;
                    }
                }
            }
        }

        // If no changes were made in this iteration, no further progress can be made
        if !changed {
            if DEBUG {
                println!(
                    "No changes made in iteration {}. Terminating constraint propagation.",
                    iterations
                );
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

// Find the cell with the fewest candidates (minimum remaining value heuristic)
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

// Backtracking solver with in-place assignments
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

        // Parse the line into a 9x9 grid
        let mut puzzle = [[0u8; 9]; 9];
        let mut valid = true;

        for (idx, ch) in trimmed.chars().enumerate() {
            let row = idx / 9;
            let col = idx % 9;

            if ch == '.' {
                puzzle[row][col] = 0; // Represent empty cells with 0
            } else if ch.is_digit(10) {
                let digit = ch.to_digit(10).unwrap() as u8;
                if digit >= 1 && digit <= 9 {
                    puzzle[row][col] = digit;
                } else {
                    eprintln!(
                        "Warning: Line {} contains invalid digit '{}'. Skipping.",
                        line_num + 1,
                        ch
                    );
                    valid = false;
                    break;
                }
            } else {
                eprintln!(
                    "Warning: Line {} contains invalid character '{}'. Skipping.",
                    line_num + 1,
                    ch
                );
                valid = false;
                break;
            }
        }

        if valid {
            puzzles.push(puzzle);
        }
    }

    Ok(puzzles)
}

fn main() {

    verify_digit_masks();

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

    println!(
        "Found {} puzzle(s) in the file '{}'.\n",
        puzzles.len(),
        filename
    );

    let total_start = Instant::now();


    puzzles.par_iter().for_each(|puzzle| {
        let mut matrix = init_sudoku_matrix(*puzzle);
        solve_sudoku(&mut matrix);

    });


    let total_duration = total_start.elapsed();
    println!(
        "\nTime for solving Puzzle {}: {:.6} seconds\n",
        puzzles.len(),
        total_duration.as_secs_f64()
    );
}

