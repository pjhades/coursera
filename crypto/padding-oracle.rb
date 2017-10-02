require 'net/http'
require 'parallel'

def from_hex_str(s)
    s.each_char.each_slice(2).map{|x, y| (x+y).to_i(16)}
end

def to_hex_str(c)
    c.map{|x| x.to_s(16).rjust(2, '0')}.join
end

def xor(a, b)
    a.zip(b).map{|x, y| x^y}
end

def send_req(guess)
    uri = URI('http://crypto-class.appspot.com/po?er=' + to_hex_str(guess))
    resp = Net::HTTP.get_response(uri)
    resp.code.to_i
end

# Generate a padding block with length `len` starting
# at `idx` that contains byte `byte`.
def get_padding(idx, len, byte)
    p = [0] * len
    p[idx*16...(idx+1)*16] = [0] * (16 - byte) + [byte] * byte
    p
end

def main
    cipher = from_hex_str('f20bdba6ff29eed7b046d1df9fb70000' +
                          '58b1ffb4210a580f748b4ac714c001bd' +
                          '4a61044426fb515dad3f21f18aa577c0' +
                          'bdf302936266926ff37dbf7035d5eeb4')
    n_blocks = cipher.length / 16

    plain = [0] * cipher.length

    (0 ... n_blocks - 1).reverse_each do |blk_idx|
        len = (blk_idx + 2) * 16
        guess = [0] * len

        15.downto(0).each do |byte_idx|
            # generate a padding
            p = get_padding(blk_idx, len, 16 - byte_idx)

            result = Parallel.map(0 ... 256, in_processes: 4) do |guessed_byte|
                # prepare the guess
                g = guess.clone
                g[blk_idx * 16 + byte_idx] = guessed_byte
                # try cipher XOR guess XOR padding
                guessed_cipher = xor(xor(g[0...len], p), cipher[0...len])
                send_req(guessed_cipher).to_i
            end

            # If we got 404 we would be sure that such guessed byte
            # was correct. Otherwise if we got 200 but no 404, although
            # the guess was cancelled with padding under XOR, we should
            # know that this byte was the only possible correct one.
            byte = result.index(404)
            if !byte
                byte = result.index(200)
            end
            guess[blk_idx * 16 + byte_idx] = byte
            plain[blk_idx * 16 + byte_idx] = byte

            puts "blk #{blk_idx} byte #{byte_idx}"
            puts to_hex_str(plain)
        end
    end

    puts plain[0...n_blocks*16].map{|x| x.chr}.join
end

main
