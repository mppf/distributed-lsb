// use rayon::prelude::*;
#![allow(non_snake_case)]
use rand::distr::StandardUniform;
use rand::prelude::*;
use rand_pcg::Pcg64;
use std::time::Instant;

use lamellar::array::prelude::*;
use lamellar::array::Distribution;

mod prefix_sum_impl;
use prefix_sum_impl::exclusive_prefix_sum;

const RADIX: u64 = 16; // TODO set to 16
const N_DIGITS: u64 = 64 / RADIX;
const N_BUCKETS: u64 = 1 << RADIX;
const COUNTS_SIZE: usize = N_BUCKETS as usize;
const MASK: u64 = N_BUCKETS - 1;

#[lamellar::AmData(Default, Debug, ArrayOps, PartialEq, PartialOrd)]
struct SortElement {
    key: u64,
    val: u64,
}

fn get_bucket(x: &SortElement, d: u64) -> usize {
    return ((x.key >> (RADIX * d)) & MASK) as usize;
}

// shuffle elements from A into B
fn global_shuffle(
    A: &ReadOnlyArray<SortElement>,
    B: &AtomicArray<SortElement>,
    digit: u64,
    world: &LamellarWorld,
    n_per_task: usize,
) {
    let my_pe = world.my_pe();
    let num_pes = world.num_pes();
    let n_tasks_per_pe = world.num_threads_per_pe();
    let n_tasks_total = num_pes * n_tasks_per_pe;

    let counts_f = AtomicArray::<i64>::new(
        world.team(),
        COUNTS_SIZE * n_tasks_per_pe * num_pes,
        Distribution::Block,
    );
    let counts = counts_f.block();

    // 'counts' stores the counts in the global order
    // that will be scanned, sorted by the digit value, and then by
    // the global task index (pe and then task within)
    //
    // t0d0, t1d0, t2d0, ...
    // t0d1, t1d1, t2d1, ...
    // ...
    //
    // that is, d*n_tasks_total + my_pe*n_tasks_per_pe + my_task

    // get a fresh reference to the same counts array
    let counts_clone = counts.clone();
    let _ = A
        .local_chunks(n_per_task)
        .enumerate()
        .for_each(move |(tid, task_slice)| {
            // compute the count for this task / chunk
            // note, vec! allocates a fixed-sized array on the heap
            let mut this_task_counts = vec![0; COUNTS_SIZE];
            for elt in task_slice {
                this_task_counts[get_bucket(elt, digit)] += 1;
            }
            // store the task count into the global count
            let mut indices = vec![0; COUNTS_SIZE];
            for i in 0..COUNTS_SIZE {
                let glob_count_idx = i * n_tasks_total + my_pe * n_tasks_per_pe + tid;

                indices[i] = glob_count_idx;
            }

            let _ = counts_clone.batch_store(indices, this_task_counts).spawn();
        })
        .spawn();
    world.wait_all();
    world.barrier();

    let counts_lck = counts.into_local_lock().block();

    // compute the exclusive prefix sum of the counts

    // compute the prefix sum of the counts
    exclusive_prefix_sum(&counts_lck, &world);

    let starts = counts_lck.into_read_only().block();

    // shuffle the data in A to B based on the counts
    // get fresh references to B and starts arrays
    let B_clone = B.clone();
    let starts_clone = starts.clone();
    let _ = A
        .local_chunks(n_per_task)
        .enumerate()
        .for_each_async(move |(tid, task_slice)| {
            // and get fresh references again
            let B_ = B_clone.clone();
            let starts_ = starts_clone.clone();

            async move {
                let mut this_task_starts;

                // load the start positions for this task / chunk
                {
                    let mut indices = vec![0; COUNTS_SIZE];
                    for i in 0..COUNTS_SIZE {
                        let glob_count_idx = i * n_tasks_total + my_pe * n_tasks_per_pe + tid;

                        indices[i] = glob_count_idx;
                    }
                    this_task_starts = starts_.batch_load(indices).await;
                }

                // store the elements in the appropriate place
                // based upon the task starts
                {
                    let mut indices_iter = task_slice.iter().map(|elt| {
                        let r = &mut this_task_starts[get_bucket(elt, digit)];
                        let idx = *r;
                        *r += 1;
                        idx as usize
                    });
                    let _ = B_
                        .batch_store(
                            &mut indices_iter as &mut dyn Iterator<Item = usize>,
                            task_slice,
                        )
                        .await;
                }
            }
        })
        .spawn();
    world.wait_all();
    world.barrier();
}

// Sort the data in A, using B as scratch space.
fn my_sort(
    mut A: ReadOnlyArray<SortElement>,
    mut B: AtomicArray<SortElement>,
    world: &LamellarWorld,
    n_per_task: usize,
) -> (ReadOnlyArray<SortElement>, AtomicArray<SortElement>) {
    assert!(N_DIGITS % 2 == 0);
    for digit in (0..N_DIGITS).step_by(2) {
        global_shuffle(&A, &B, digit, world, n_per_task);
        let B_ = B.into_read_only().block();
        let A_ = A.into_atomic().block();

        global_shuffle(&B_, &A_, digit + 1, world, n_per_task);
        A = A_.into_read_only().block();
        B = B_.into_atomic().block();
    }
    (A, B)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let world = lamellar::LamellarWorldBuilder::new().build();
    let my_pe = world.my_pe();
    let num_pes = world.num_pes();
    let mut n: usize = args
        .get(1)
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or_else(|| 64);
    // TODO: try multiplying n_tasks_per_pe by 2
    let n_tasks_per_pe: usize = world.num_threads_per_pe();
    let n_per_task: usize = usize::div_ceil(n, num_pes * n_tasks_per_pe);
    let n_per_pe: usize = n_tasks_per_pe * n_per_task;
    let n_tasks_total = num_pes * n_tasks_per_pe;
    n = n_per_task * n_tasks_total;
    if my_pe == 0 {
        println!("hello from pe {} of {}", my_pe, num_pes);
        println!("there are {} tasks per pe", n_tasks_per_pe);
        println!(
            "n is {} n_per_pe is {} n_per_task is {}",
            n, n_per_pe, n_per_task
        );
    }

    let A_f = LocalLockArray::<SortElement>::new(world.team(), n, Distribution::Block);
    let B_f = AtomicArray::<SortElement>::new(world.team(), n, Distribution::Block);
    let A = A_f.block();
    let B = B_f.block();

    let time_rng = Instant::now();

    let glob_start = n_per_pe * my_pe;

    let _ = A
        .write_local_chunks(n_per_task)
        .block()
        .enumerate()
        .for_each(move |(tid, mut task_slice)| {
            let seed = (42 + my_pe * n_tasks_per_pe + tid) as u64;
            let mut rng = Pcg64::seed_from_u64(seed);
            for (i, elt) in task_slice.iter_mut().enumerate() {
                let idx = (glob_start + tid * n_per_task + i) as u64;
                let r = rng.sample(StandardUniform);
                *elt = SortElement { key: r, val: idx };
            }
        })
        .spawn();
    A.wait_all();
    world.barrier();

    if my_pe == 0 {
        println!("elapsed rng time {}", time_rng.elapsed().as_secs_f64());
    }

    /*println!("Input for sort");
    A.print();*/
    let A = A.into_read_only().block();
    world.barrier();
    let time_sort = Instant::now();

    let (A, _B) = my_sort(A, B, &world, n_per_task);

    world.barrier();

    /*println!("Output from sort");
    A.print();*/

    if my_pe == 0 {
        let time_sort = time_sort.elapsed().as_secs_f64();
        println!("Sorted {} values in {}", n, time_sort);
        println!("That's {} M elements sorted / s",
                 (n as f64 / 1000000.0) / time_sort);
    }

    /* BEGIN_IGNORE_FOR_LINE_COUNT (verification code) */

    //roughly check that the data is sorted
    A.local_chunks(n_per_task)
        .for_each(move |task_slice| {
            let mut failed = vec![];
            for i in 1..task_slice.len() {
                let prev = task_slice[i - 1].key;
                let curr = task_slice[i].key;
                if prev > curr {
                    failed.push((i, prev, curr));
                }
            }
            if failed.len() > 0 {
                println!(" [{}]  {:?}: ", my_pe, failed.len());
            }
        })
        .block();

    /* END_IGNORE_FOR_LINE_COUNT */
}
