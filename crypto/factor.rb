require 'gmp'
require_relative 'lib.rb'

def challenge1
    n = GMP::Z.new('17976931348623159077293051907890247336179769789423065727343008115' + \
                   '77326758055056206869853794492129829595855013875371640157101398586' + \
                   '47833778606925583497541085196591615128057575940752635007475935288' + \
                   '71082364994994077189561705436114947486504671101510156394068052754' + \
                   '0071584560878577663743040086340742855278549092581')
    r = n.sqrt
    a = r ** 2 == n ? r : r + 1
    x = (a ** 2 - n).sqrt

    [a - x, a + x]
end

def challenge2
    n = GMP::Z.new('6484558428080716696628242653467722787263437207069762630604390703787' + \
                   '9730861808111646271401527606141756919558732184025452065542490671989' + \
                   '2428844841839353281972988531310511738648965962582821502504990264452' + \
                   '1008852816733037111422964210278402893076574586452336833570778346897' + \
                   '15838646088239640236866252211790085787877')
    r = n.sqrt
    gen = 0.step
    (0 ... 2**20).each do |i|
        a = r + i
        next if a ** 2 < n
        x = (a ** 2 - n).sqrt
        if (a - x) * (a + x) == n
            return [a - x, a + x]
        end
    end
end

def challenge3
    # let B = (3p + 2q)/2, we know that this is odd
    # so we deal with 2B = 3p + 2q, which is the mean
    # of 6p and 4q, all the remaining reasoning is the
    # same as challenge #1
    n = GMP::Z.new('72006226374735042527956443552558373833808445147399984182665305798191' + \
                   '63556901883377904234086641876639384851752649940178970835240791356868' + \
                   '77441155132015188279331812309091996246361896836573643119174094961348' + \
                   '52463970788523879939683923036467667022162701835329944324119217381272' + \
                   '9276147530748597302192751375739387929')
    r = (n * 24).sqrt
    b = r ** 2 == n * 24 ? r : r + 1 # compute 2B
    x = (b**2 - 24*n).sqrt
    p = b - x
    q = b + x
    if p.divisible?(6) && q.divisible?(4)
        return [p/6, q/4].sort
    end
    if p.divisible?(4) && q.divisible?(6)
        return [p/4, q/6].sort
    end
end

def challenge4
    n = GMP::Z.new('17976931348623159077293051907890247336179769789423065727343008115' + \
                   '77326758055056206869853794492129829595855013875371640157101398586' + \
                   '47833778606925583497541085196591615128057575940752635007475935288' + \
                   '71082364994994077189561705436114947486504671101510156394068052754' + \
                   '0071584560878577663743040086340742855278549092581')
    e = GMP::Z.new(65537)
    cipher = GMP::Z.new('220964518674103817763065611348834180174100697878928310717318' + \
                        '391436761356001205380042823296504735094243439462197515122564' + \
                        '658399679428894607645420405815647489880137348641204523252293' + \
                        '201764879166664029975091887299716905260832220677716000193292' + \
                        '608700095799937240774589677736978175712672299511486629596279' + \
                        '34791540')
    # factorize the modulus
    p, q = challenge1
    # compute phi(N)
    phi_n = n - p - q + 1
    # get private key d
    d = e.invert(phi_n)

    padded = cipher.powmod(d, n).to_s(16)
    padded = '0' + padded if padded.length % 2 == 1

    message = padded[(padded.index('00')+2) .. -1]
    to_ascii_str(from_hex_str(message))
end

def main
    puts challenge1.min
    puts challenge2.min
    puts challenge3.min
    puts challenge4
end

main
