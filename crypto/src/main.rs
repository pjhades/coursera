extern crate crypto;
extern crate openssl;

mod aes;
mod mac;

fn main() {
    aes::solve();
    mac::solve();
}
