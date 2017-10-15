def from_hex_str(s)
    s.each_char.each_slice(2).map{|x, y| (x+y).to_i(16)}
end

def to_hex_str(c)
    c.map{|x| x.to_s(16).rjust(2, '0')}.join
end

def to_ascii_str(m)
    m.map{|x| x.chr}.join.inspect
end
