use std::str;

pub fn hex_to_u8(s: &&str) -> Vec<u8> {
    s.as_bytes().chunks(2).map(|chunk| {
        u8::from_str_radix(str::from_utf8(&chunk).unwrap(), 16).unwrap()
    }).collect()
}
