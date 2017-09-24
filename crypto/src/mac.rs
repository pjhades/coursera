use openssl::sha::Sha256;
use std::fs::File;
use std::io::Read;

type Sha256Hash = [u8;32];

fn video_hash(path: &str) -> Sha256Hash {
    let mut block = [0u8;1024];
    let mut blocks: Vec<Vec<u8>> = Vec::new();
    let mut file = File::open(path).unwrap();

    while let Ok(n_read) = file.read(&mut block) {
        if n_read == 0 {
            break;
        }
        blocks.push((&block[..n_read]).to_vec());
    }

    blocks.iter().rev().fold(None, |last_hash: Option<Sha256Hash>, block: &Vec<u8>| {
        let mut sha = Sha256::new();
        sha.update(block.as_slice());
        if let Some(h) = last_hash {
            sha.update(&h);
        }
        Some(sha.finish())
    }).unwrap_or([0u8;32])
}

fn print_hash(hash: &Sha256Hash) {
    for x in hash.iter() {
        print!("{:02x}", x);
    }
    println!("");
}

pub fn solve() {
    print_hash(&video_hash("/Users/pjhades/code/lab/try/video-test.bin"));
    print_hash(&video_hash("/Users/pjhades/code/lab/try/video.bin"));
}
