#![allow(non_snake_case)]
use rand::distr::StandardUniform;
use rand::prelude::*;
use rand_pcg::Pcg64;
use std::time::Instant;

use lamellar::array::prelude::*;
use lamellar::array::Distribution;

mod prefix_sum_impl;
use prefix_sum_impl::exclusive_prefix_sum;

const RADIX: u64 = 16;
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

fn calc_global_index(
    bucket: usize,
    my_pe: usize,
    my_task: usize,
    num_pes: usize,
    n_tasks: usize,
) -> usize {
    return (bucket * num_pes * n_tasks) + (my_pe * n_tasks) + my_task;
}

// shuffle elements from A into B
fn global_shuffle(
    A: &UnsafeArray<SortElement>,
    B: &UnsafeArray<SortElement>,
    digit: u64,
    world: &LamellarWorld,
    n_per_task: usize,
) {
    let my_pe = world.my_pe();
    let num_pes = world.num_pes();
    let n_tasks_per_pe = world.num_threads_per_pe();
    let n_tasks_total = num_pes * n_tasks_per_pe;

    let counts_f = UnsafeArray::<i64>::new(
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

    //println!("global shuffle 1");

    unsafe {
        // get a fresh reference to the same counts array
        let counts_clone = counts.clone();
        let _ = A
            .local_chunks(n_per_task)
            .enumerate()
            .for_each(move |(tid, task_slice)| {
                //println!("tid {:?} got chunk {:?}", tid, task_slice);
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
        //A.wait_all();
        //counts.wait_all();
        world.wait_all();
        world.barrier();
    }

    /*println!("counts is");
    counts.print();
    unsafe {
        println!("counts sum is {:?}", counts.sum().block());
    }*/

    //println!("global shuffle 2a");

    let counts_lck = counts.into_local_lock().block();

    //println!("global shuffle 2b");

    // compute the exclusive prefix sum of the counts

    {
        // copy the counts (for use in computing the exclusive prefix sum)
        // let copy_counts = LocalLockArray::<i64>::new(world.team(), counts_lck.len(), Distribution::Block).block();
        // copy_counts.local_iter_mut().zip(counts_lck.local_iter()).for_each(move|(copy,count)| { *copy = *count; }).block();

        /*println!("copy_counts is");
        copy_counts.print();*/

        //println!("global shuffle 2c");

        // compute the prefix sum of the counts
        exclusive_prefix_sum(&counts_lck, &world);

        //println!("global shuffle 2d");

        /*println!("inclusive prefix sum is");
        counts_lck.print();*/

        // subtract the count from each entry in the inclusive
        // prefix sum to get the exclusive prefix sum
        // counts_lck.local_iter_mut().zip(copy_counts.local_iter()).for_each(move|(start,count)| { *start -= *count; }).block();

        // println!("exclusive prefix sum is");
        // counts_lck.print();

        /*array_A.local_iter().zip(array_B.local_iter()).for_each(move|(elem_A,elem_B)| println!("PE: {my_pe} A: {elem_A} B: {elem_B}")).block();
        counts_lck.dist_iter_mut().for_each(move |elt|{
            *elt = *elt -
        }).block();
        counts_lck.write_local_chunks
        let copy_array = array.dist_iter().collect::<AtomicArray<usize>>(Distribution::Block).block();*/
    }

    //println!("global shuffle 3");
    //
    let starts = counts_lck.into_unsafe().block();

    /*println!("starts is");
    starts.print();
    println!("global shuffle 4");*/

    // shuffle the data in A to B based on the counts
    unsafe {
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
                    //println!("tid {:?} got chunk {:?}", tid, task_slice);
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
                        // let size = task_slice.len();
                        // let mut indices = vec![0; size];
                        // for i in 0..size {
                        //     let elt = &task_slice[i];
                        //     let r = &mut this_task_starts[get_bucket(elt, digit)];
                        //     indices[i] = *r as usize;
                        //     *r += 1;
                        // }

                        // //println!("tid {:?} chunk {:?} indices {:?}", tid, task_slice, indices);

                        // let _ = B_.batch_store(indices, task_slice).await;

                        /* it'd be cool if we could do this */
                        let mut indices_iter = task_slice.iter().enumerate().map(|(i, elt)| {
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
                            .spawn();
                    }
                }
            })
            .spawn();
        //A.wait_all();
        //B.wait_all();
        world.wait_all();
        world.barrier();
    }
    //println!("global shuffle 5");
}

// Sort the data in A, using B as scratch space.
fn my_sort(
    A: &UnsafeArray<SortElement>,
    B: &UnsafeArray<SortElement>,
    world: &LamellarWorld,
    n_per_task: usize,
) {
    assert!(N_DIGITS % 2 == 0);

    for digit in (0..N_DIGITS).step_by(2) {
        global_shuffle(A, B, digit, world, n_per_task);
        global_shuffle(B, A, digit + 1, world, n_per_task);
    }
}

// TODO: batch_load, batch_store
// rayon scope / spawn
// par_chunks_mut
//
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

    let A_f = UnsafeArray::<SortElement>::new(world.team(), n, Distribution::Block);
    let B_f = UnsafeArray::<SortElement>::new(world.team(), n, Distribution::Block);
    let A = A_f.block();
    let B = B_f.block();

    let time_rng = Instant::now();

    // set the input
    unsafe {
        /*input_array
        .dist_iter_mut()
        .enumerate()
        .map_with(rng, |rng, | rng.sample(Standard))
        .for_each(move |elem| *elem = (rand, idx));*/

        /* bad: same values for different tasks
        let rng = Pcg64::seed_from_u64((42 + my_pe) as u64);
        let start = (n_per_pe * my_pe) as u64;
        println!("start is {}", start);
        A.mut_local_data()
         .par_iter_mut()
         .enumerate()
         .map_with(rng, |rng, (i,elt)|
                   (elt, rng.sample(StandardUniform), i))
         .for_each(move|(elt, r, i)|
                   *elt = SortElement {key:r,
                                       val: (i as u64) + start});*/

        let glob_start = n_per_pe * my_pe;

        /*
        let loc_A_ptr = A.mut_local_data().as_mut_ptr(); // `*mut SortElement` cannot be sent between threads safely
        //let loc_A = A.mut_local_data(); cannot borrow `*loc_A` as mutable more than once at a time

        rayon::scope(move |s| {
            for tid in 0..n_tasks {
                s.spawn(move |_| {
                    let loc_task_start = tid * n_per_task;
                    let loc_task_end = loc_task_start + n_per_task;
                    //let loc_A_ptr = loc_A.as_mut_ptr();
                    let task_slice =
                        slice::from_raw_parts_mut(loc_A_ptr.add(loc_task_start),
                                                  loc_task_end - loc_task_start);
                    let seed = (42 + my_pe*n_tasks + tid) as u64;
                    let mut rng = Pcg64::seed_from_u64(seed);
                    for (i, elt) in task_slice.iter_mut().enumerate() {
                        let idx = (glob_start + i) as u64;
                        let r = rng.sample(StandardUniform);
                        *elt = SortElement {key:r, val: idx};
                    }
                });
            }
        });*/
        /*
          let loc_A = A.mut_local_data();
          loc_A.par_chunks_mut(n_per_task)
               .for_each(|slice| {
                           println!("got chunk {:?}", slice);
                         });
        */

        let _ = A
            .local_chunks_mut(n_per_task)
            .enumerate()
            .for_each(move |(tid, task_slice)| {
                //println!("tid {:?} got chunk {:?}", tid, task_slice);
                let seed = (42 + my_pe * n_tasks_per_pe + tid) as u64;
                let mut rng = Pcg64::seed_from_u64(seed);
                for (i, elt) in task_slice.iter_mut().enumerate() {
                    let idx = (glob_start + tid * n_per_task + i) as u64;
                    let r = rng.sample(StandardUniform);
                    *elt = SortElement { key: r, val: idx };
                }
                //println!("tid {:?} ->  chunk {:?}", tid, task_slice);
            })
            .spawn();
        A.wait_all();

        // try local_iter_mut / local_chunks_mut
    }

    world.barrier();
    // A.print();

    if my_pe == 0 {
        println!("elapsed rng time {}", time_rng.elapsed().as_secs_f64());
    }

    /*println!("Input for sort");
    A.print();*/

    world.barrier();
    let time_sort = Instant::now();

    my_sort(&A, &B, &world, n_per_task);

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

    // roughly check that the data is sorted
    unsafe {
        A.local_chunks(n_per_task)
            .enumerate()
            .for_each(move |(tid, task_slice)| {
                let mut failed = false;
                for i in 1..task_slice.len() {
                    let prev = task_slice[i - 1].key;
                    let curr = task_slice[i].key;
                    failed |= prev > curr;
                }
                if failed {
                    println!("pe {} task {} failed", my_pe, tid);
                }
            })
            .block()
    }

    /* END_IGNORE_FOR_LINE_COUNT */
}
