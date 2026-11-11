#!/usr/bin/env python3
"""Offline additive experiment: existing HiGGs Beta-H1 two-hop mean + HOG RW/SPD.
No Git, network, real-data reads, training or sampling.
Accepts either the synchronized original mean-capable H1 or the first HOG patch.
Existing mean config, two-hop modules, Beta primitives, H2, Glue and metrics remain unchanged.
HOG code attribution and CC BY-NC 4.0 notice are included in the payload.
"""
import argparse
import base64
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import zlib

BASE_H1_SHA = "4d64a3cb591ffd530f13c92ac3d7a8d19aa3caa5a53ff00d19f19470686ec367"
H1 = "dgd/diffusion_model_beta_h1.py"
STACK_CFG = "configs/model/beta_h1_twohop_mean_hog_rw.yaml"
ALLOWED = {
    H1, STACK_CFG, "dgd/diffusion/hog_rw_features.py",
    "configs/model/beta_h1_hog_rw.yaml", "THIRD_PARTY_NOTICES_HOG_RW.md",
    "tests/test_h1_hog_rw.py",
}
STATE = ".tmp/h1_twohop_mean_hog_rw_20261022"
PAYLOAD_SHA = "06796b82ab1a1d0475278f346f28b1eea56d068730e5a054bc2a7ab584e467df"
PAYLOAD = "eNrtfQ1z2ziS6F/heqtuqQlNS/JHEs/q6jIZTeK6TJIXe95my/ZxKBGSuaZILUnZ0eZ8v/11Nz4IgKAkOzNv76ous5uIJLrRaDQa3Y0G8HVvlmas2jv1vu5dvD379GP08dWni79G7z9cnL0en0dvP7yJPv0lXCRQYu+PHjzu/5jOZt6nvxycf/zRS/Plqvaq1WRaLJZFzvL6Kr/Kz4tVOWWn3k1dL6vTg4N5Wt+sJiGUOfhrukjz+c3NgcR0MMmKycEiTvODRZGwrDpY1WlWhcv1Vf6hTOdpHmce0ui9SWsPC596vx4nz4ezo8MXk8FgOExOZv3jyck0fsleHk8Gh8cnLwcz+MFms19DDUm8qm+Ksjr1OA3e21UMf8d54l0U2Tz2fkjLJM4Cb9gfngDcx3jJylPVYviVzm9Yuf+hTFjpvVmlCUs8/LKq0iL3ZgW8LOPljfeG5ayMa3gZeGev332SCK/yixvmxUm8rAHy1zmro/I+mrG4/tWbrfIpQgBDvV+TeXKQSMQHN8VclluVDBnzq5dWV3mWTlleAaZVjgS9LqFAese818ViUeSV96quy3SyQqz774scX7NymgIjjsK+d5bXrMyJyjg7vcplV00FminHEhbl/EDUVB1M1vv59ADAD67yn1ZZ5okvTU/vCJ6xeZxNobuv8l+WVQ1ACy8v6vQJQvMOxPT9+RjZ+7pYrqGzb2rPn/aI65u7GvrkJ+i1acMZTipCpPnfV2mZsiqA73kdT2uvht4rLGEKsVPTCr4Bxv1lXNZrTw0F6CZslVcyq68QU7UqywIesbJlWfyNTes/Nb0aeu8Lj+UJVMEWiGqydtaPVUyzOF2whEDu47KM8xroJoF7fQPtZiDy1Ma8rogHIKqLuK6h5u+9Ylnvg8zBPyAJ1ffw3WNfSDQybxFXt0geF9ny+6scWDGNa8blBnhUFxwgrWpic/rmTeX9wOp4/+3AS1hepBUrQw/lHshKisX+fZzdesvinpXQWCTmH6ws9qfAidpDhRJnc2hjfbPw4hKAAFWaY9uaoQMdlrEYmr6ME+IeYLmCMnO2n6TxHOXZm0K7c9AloTeWtE3XAHZQLYHPJZTgiksOKrMyPlRTYm7sQfG8mkG17AsohBTZGHigHuET9nTGataoRRCn82lRsvcMu30JvUmsApRntZcUjAtEmk+zVcLMHk3YvGTsYAIP5Xo/Tv4WgyRM14JS2aLAe/36p+AqVw1RigLrbh4mZQocAQZgb1CPr7IYoLOiqlCo3w5JFN5kKxiD2HpQQCQswGscR8RYb7maZGl1A4LbEuqYy51XzLyc3e8vGMgj9ENe3LGsXgeEPS+8uVKG+39fxVkK42MOXCbWAiUlqsIJA/qYxl8U3r3A2wNxm6Xzik8MBxNoSXQziLhCDNfxInNPSvp0pLovEFQDsTCwQPdQC6W0QYV/9M7hVcb2ZzDYC727T6HnCLpKgYt1nLNiVWVrkMcEhi7oOhjF2MID3oUe0An4EjaLoXQFytXz9j1BPX+IKpbNIpQz3hgqA8NqkrHkFEjGTvE80PoJW9Y3p94RFq3vC2j70iw7i7OKCksBOfUGJ/ic4yDP0n8Q70+9BYtzfA0KYhFVNSiqaHrDprfLIsUG5qDNsQ5oTqTao7Aj6aClYeTP1/Ay/cKSTR0EhAKdEVbZ7qxz0KW30Ok6e/nIo+GgdAlCe4BpH1CRMMXLJShE7GzRz9RnoPO43HqgixBBI7JS8qHJyUHJ7kDlMFS1U1ZVOFjYXZyt5PB095belN+652TZjo6y+/O37bhtJgb21RX9t93i49oQed/WhyQYpFA1c4cGvzCDZmWx2GaQKWw+zuk9zVp5qmGpjI7GptzVpAyfZGcA5z7vv+Nz+/4ZzIx1OkvRvHz9ev+Hv+6/f70PZtETrLBzUDcbzXayS0Ho70Du8ikj2pSV47YRDqp6nbHAAxalYPYIC4EP9ZU0bGlEarYCTf1XuTQZ7kscrmLqV1Mcn/e9j/8xDMIwDD7+h3/7bNDzHGaA1NB8btLNAD4t8/kXZ1iYJclAugdRAHLQUKvAMqpZVYM9Bu8SUCjYdJxUUFvIWatrKggaMVZTOfBQCaE5rfNBgnwkSZ4WWcbIkq/CeDL1UhgjZe39DNyg1ohnmF+mNwjE+yl69e7Nh09nF29/9kZeM+ROpcCeSuE81UYRr5U0lycGsLDifN5XI7D6WY+UjudJ/QYVfL3aE0roCob5T6gpAqhV6ix8efTAodKZ6HfiOaAT2PAPdMqqzBViBUAGDkgX57qgJZAs6OkYYjAPvYv1ko3BFi79qz2uLnhrvMWqgsEJIguGKGffXo8Dr/LbvLjPoS0Vq0UNPVDO+CTp6SmCROlWxf8X1L+oeXa194vAKSoXrAS3uEJhSnyBpfegqABNCRUBEQkME1Vv4H33Hb0QZDV01NBQnwNdal1w3ZMm1aQoso3sEZOpAFUMQjhFVauipl+bmmDqQJnGnwPvzyPPVRjfHw43cU0RJKGaLiPngM3BagbVcTkIANO1xjiSHF4pF+J/oxER5kU0L+PE73Gx1qTdv4WJjC0D9CoqFoF5LCWJj0Dl5atxiqN4VdM4/V5UWcEk5mVxLt0GUjXKV5BDmUi8j8CKwM69vJYDaI5dLWsPq9XC34eG3TK2TNJFNbqA+Vw08NWPelHvwPMR+pk3YPsvekYFIbdq/Fc/iveoriNkWolaWbTaGDT3gJtza7JY+ALP5f7gOvAUFkcN5X1TM4wnhaRCe0yhGZwCGmjOaKDRKRgmIaSX5XNMASjTGYy8UZ8gB6Mh/TscHbpQNA8hzA+LVc18gAMQWWG9WEbURo49xOHnc4ou+6IvqmUSpdB9I88XxUFW+z3qE514LAdK66aoFfF5HspYC0zM8BEMs9pvuCZQg7ewWkQwV6C3NOK9gB3YC+vC55hmWRHXroqaB72Jh9BEaKg1BBQ3Ag1s05DACZYlkWWx+dxrRGlDPydhEZYLlB1qjpVXJBH7RQ4+DHfZpYfvFasazbvPo8sfgvfBLQjDmP+Eh2eDa2OMgKZpqg2hFQvvD6AyULFo76ubeMkuB9f4rfV6eL1ZvzQAXLXcxHfMI1CPU3Wt6z3V8DBBFYg1ioECCpLrO1mA43CRdLqNJoXE0L9ET0ONNSGbUy5qCnvGlb8fesZEyulPq1mapyBHDbW9MM4yv7eZ1Pc4BSGg14QTlijSCXpLjSdlMhHMWq0ib4RDy/tPz3o36O1EQlOJCt+g5zfFUJRgfkOcogOndocM8Y/gtqU49lVHXJ4GHvwP7ZNr71/M9/gSvkolLgNEUjLYmvk5Tip3YBOPtCr5G/iCkjRqxKh3SdVolEQYegKEnKx/8f5LVsLLELv0xpDe8Hvedxq4o88Zhkt8GtAIRWGMZVExn7TI7lwvGQYyMcSVY/QRfk7R72rzvLwnHQTE6tOulEuiQpUk9QwtaAbUKq/+vmLsHwwmxUYpcm1oaXrUhGbzXdBKQxJVDq81ImMxEp664bO+lo4w9DSPyVMswAzWc813XsfQexf3hYfqMLTM+SiarVDBRpG05MFWKLhRUWEp8XZaoFf5GA+gqER59FGydCLLfYRHDfFyTdIAkzO4mzmZLZW3zGxHQn+CKQ4L5bmGhhwJUaFiYEisa/hJhSQdYkoEqrM1TkDLG+rnKuDvMQqM/tI0ziI9BIYRiloWErEX3kVVjE6V+JLmFI2JKoCHv2/SWS2+ZAX49FVUF9EKFBfgLibxJMW4YSDlgoNypGgSCmlDJDAzYnA43wBfpfNFAXIXZ8ubOJD+hFkUwSU+bBCS3RPsWzB0iyvOPBiVKQgifyVZh7L2dvCuqCTHefiBD2GMxLKSy64EIBG9aL5aYDx4xNWM7GjxTswvrZ61zAOrU22nM7Cmq6DDyuBcwP/IMPK0YUZNwIarMfZ24C+z8J2U25+LZJUx0w4RC2XkLrwdUPwBe2Ltkbx59ze46of2NS3IwIzxdujxmpW88ZiWNmxJ7uJFCqYNutIy6ipMlopGO37g75RvDCMdJ8oo8jG+F3jTGVhkSVzHYNjCl1kBTDE6m5zroFHC6g/JORBsFvTu0mqlonwgX0XWjYJ9gaoU1wWCpMBQlvlWnwWqFWhZvxeqhmiuAEwsfNaBZoWiVeFND+2fQdg/NUlwTCdbOrqZYxTu0UDNK4KAqi6peu7iwxQDePXw5NUemkQ8RLnXI9rU41YKWzxUPY98QdmvPL2yESEmlf+9J4XDhJfLMay+ZwwcxKo6SNY5SNYUMUlHWmAnmqU4UW+AFIXQXJA1+Nt6D1gY/xDy5ZGMXlmlLqAIONEa0xpVjXqv6lkAurii/6lLr00BztsJd3CNcvSlok8WiC3XOLVbrywIh8ijmdR+a8GZ4g8g5gu71ea4wAaZb/RuoeB1hO66CtsYrW++a8zlPtEGKK2A3Sef4WUknCrenVrhy6u9zxiVsWDGW2DGDpj1Fpi1CYPj0a4KBtzwKQONlnthoE1XZUnrgzjPY0iWLxKTl6kGobD4ue2FM1plDZw/UsSYFnGmoljJwMKoFMIbMG4u82IfoQPCcS1WOdX40NHxWZSTQQtMuFAsEMrYD8WCyIoBBbFC30Tg9zJQF2XoEiDREd4+6FCnrLgLENut6Ru9EfNNS1OK76gZLVzm1A2o7FiwhYl/NhFJcTBBjSClJRl/9PiygeQrqMOsmOLKO1jCgZejkeZ1ja3QxGXwFAeE92zUQY8WntyAY7wrDozoWOyMZE80i2wRTJ2Y6zPisfIuAG2BLl7BdAnlcY5uik/i6e0E3sCHtsVnF4puAd+cwvUP7Y6yxGVTT+mmZLOgqdmi0jh8OwAv6G2x7KbNasTjAESDLpUsRzLwih0x6mqXVmhXvMaC6kbkdkmra7mRPlJVWeovj7J4Db7IqBli8lXQJZ2j5qdV5iZNEgbmZbbkBRuk1hc3nBvGUV7TUCPtt1UqntbRLIeRO8rz8BN794vfc5cAFJ1FvvvO6iTte9ta4pyplvEUhcs0F7Vvtv4zFr6xs8Ci0CCNzzas9AVp1AJsYyRzYON7FzCjQHQXKHztAqTV0A2g9L3nstzIbXZBNu40FbzsX7vhscQu8IMWPM6KO9WPBR31c/gd6id4R/1gxSNpDmhMu8FPLhARo3OB4CcnlfUNWAk3RebqXrOADU6+B1joDkD5yZx90ZXoh31cxnDV/+cd/TQLTMancQmuH3iDa5eTwo0mFn2WalI3Wf+1NT9i/hhGL1YkAE2ow9KNWbyYJJztI5sF4hu50yghltZQXsU9w+DByMlBJCO0StqIpkVZ8kjcdlStsjYy6LKR0bdduowG2LJMi5JHUuMaNJkVSRCLCBhartDvJV/eEgiFRay84/pFxnK/+UI+st1llpAY1Ih1q1WW+b4NF/QClDHvoIVRI4zka2vbuBR2t03DYrWt+dK0bdzZNoOaVtvGnW0bO9tGn0o2B8+XldFkNZsx298R3SZMEl41NpHXHVdRzfIKBqLWRTLa/9iKqG1bK9L4JSvCFVxDHutybfFNs2yNvsOQBFjvIf5l9mrPMmdm6IRR53fhQjnAF1xTgN45bbuOXbBhA9iyobVwxB2LbkDIymVcxgsGvMRlNod/ivGXvVOK6DnjPEaYBAt2NckF7YpmAA7Haye4FdkASPONm2IrvrF3aoc8nGB8uYTHRhGG2MgfneUpVqoK0pNV7kFXDV+m4FR5Y/oHLTHw0hlOS1bHg7TmNebbXJJzfy0zLT2zL6HSMlUc9KrbdLnEpMmvhPPBCC5ylA1GvqQDKKnFp9621aBWpPJxrrBd/dtBKyFZ9IvKRT51xhzBV8OwI7imo68uGv4kPdc/XT98341Bcza+Nr8fvjcs/6/awyZkIgFXEGR7UH8STNlMkOFldWEyCm3Ad7WnsgmpF4mlKY99qBR2Z48+wmd2dCll86h8ZBhjaQIj/FSm8wYqjWjUyYeOlktAarTJq8ei2p2JlG6JCwU0yJoc/Eb5eRj10AxGXCTByDPq3aaUXC5RL3RF37y9tNWPcsrFCzeMUEGqMH/+hrFqYFdbCXgNzskD1xpJRZx6Q6eW1HM4tsabnBhU4imiaC3NuSBUMPB0u2TbKvubxoXNPkXIbvzz9HDO6Q4RH2frraDN6a7BHSeyJminMHUG9NqsNEYGjpddRwYKyQLV10grIAO0jVh+07zUTsWVtTqTcU238tMqr9PFxnwSXJDFhFlNZSBytJkMyqm9S2dLccqPGvsNwNAT0AOe3CG8w3V9Wky7WQpYsup62Ei9gcumaR6IEHNZj4RNhvYUboGW3rcRq2Lb8bfZrtX4TZzXGK06gXBzQ0dkard4z6uHvm+vDWiUbVgdaJCIkeFYr2hjcq5YUNIUVwmCGhQZAaF0Rdvd8JtG/MEZ13fNch4KE+VCC8x2+rOkhL/yfC0/04HJorZR69xXlQRuWqXoxqZNAoTOngc2t89CprFeTy3fSsulBnrdNEq8M3RNJ65uENVzmyacTry+EpuRN6D1uu6aet82ug7ElodFWi3ienpjCCTL0lmjvgV3rZ0RndW9puwxsqxo84zI43ePcXCgJuB9ZmnODAo2zBtNx9t69AmTfVuR7Th36DsWznKeFgkWtLSd/0fPF44V8K4ppF3yyXNK73edYi7aHfPo+eXbpgTRIILYJJJdTXAqDRhuTYPU1muOWwrj996qoo3HWTpNa9pjuc8Xg9yOE2LF3FHlg4Jc83hx5QLonPlcs16zBUJjy6Onr904RvudVO6SGppyPErUp95X8esPVsyFJ9OBx8h72GySZXmj3V/9HS1sCwFKGavqBsejl2nFQiR44gYKLnR64mRo4nBJX9fQdhCAk5DW+jbgaNRq3Egn1Bre3ybmcvwaVKrZ67RTlEEIlA4afeVttUMI0O/BJgRSYgC+4cc2IMWa0VeTSwTI2TT6qrEL3ocbRuQ7No+na01/VbQbtFjVFk9wLylKYrjDYBXtkd6oKeGNjxooIdH8VocB266PhrZViz7E2xBix16HIDjo0ORUvXXCjh6ff2GDOsZcd529pw0A11SlOshprm3P6bkQ2/B3TOmxmK/JhB5KsA0N03JyTfN2nG2W1hwZxRF0BtGm5s7+ch4TYHQeUEYoNNtVHNZRb2PXTqlFDQUys0ijSU6c4iQMa1FnE+PdSESUBRvU27LQoJmgiMsjXLi/d2PND1YIVG6DZ8TzRfoPXHeyu0jsXeHLdVQufJXEi7/4jjZry1e9wMvKkUwdDmmBPMxKUImLCvcA0hZTKwrFrQ/wN6fx2gbVvxmN4JsfBO9qB+XGJgmL6jpwJM8ErZyYwJHsYhJBGyhoIbMSpExw/EYVMDXwckrzqAIgacZKBqNIpxNhP5vLviaVvhPXoBeYaSQWN8U+MFWltQ1Me4/PPZOe8ePpcRNGma2/EWGiQ4lf4TSLF8tokea+kcoAdRP9nd+NbqMNMXKDjeg4vknGo/ydgG/FMfYm8L00zZ5jLhQ+gfWUc2pkgGH6P20Gutrj+pIKtxrm2Enki+osenjr/H7Yp2wAq1Fitw9u/MF6ZLv4a70t24i14wC6KqAiLjuTVyMoNNNMKHNh38wr6rU7WKAwGqXy3kRr1uYI4yJktW3tOOmha3Ou3GEh0nCMrYbKp+I7LdbNlh1jftYz9+zshrXalykolTkOBsEKA9+BjOEhe9VCq6Mp3Ww57HchRDbB1ybk1HAPM6N2qgb4AJDG6N8fGOvmS74hc2SqCDur36SxXi0zOlpDUoobZClwKPE9PpMfpsqxJGYtNll/lejALp+DffDVXfGDvVTh2LrcJGG2tg13yJcmM/wMPjCnvPfxe7SDz/KZUacYBY2Q6eNA36AohsLnCKasMf5FUxz+IBUpNKG2id4aH8pKMg9ccg56WYkeu4tzrKSqE5q+Nm+jtDpJUBrepeyejp4Y8HlDEN6kdxp5Q4HD5miSQINWVugzO8/UhUBTfoFnzxVGY8e8sePfpLFNe8eBlii5tb1N0mnQykJ9Zue1Prm9otdbW2ZtSwDEAhQ6F4UeJqWhMHTPzRb0uIEeS+jxhpm9EWTT+KI8I3TdWyOi6RjQwHXHUOA7dJqsUcssMIkm3LZY2ZLX27afXMv547t3dql6HJl9rlXdCIHuMNIGoFUJMipyhn1n1iGn4F9d6bq9EG2wKK58vWyvxbvGXoSR4ftinjfrD+wXPX48i8EUvpF+GaelfiqCxsJBTz8WQfugL3c5z0awfCpBOJ8ZcSzaxyO4zVYFJydy9+Sr8UX9/M7zteZppyv0ukQEDx1dk2S3kxKu9kAUcaVfim9gfx8b38et72v+XR8jziRRZDQWbA5jsYsRHj2pwLFcY22FdJtmvBCORGhcmLP7CLN7Kt//Dp/FqSZ0PlDfDrxzUGT42AId7wa61o2daK2j0F8LS8pCg2a9qy1yYJut95uOdZKCczdvTqARR79C+S2U38J1m9n2TtLuxT1eUpJpwW2l0xz2iniOJvzsGuFmX7Wgxwp6vBl67YJeK+i1C/pJmwPp7C3GI0R0MqTYHEq84a4cbR2UWwWpTEZHfsblnNXWFkGo9zM/IQnb33EGUtuklaP4Ek8b9FCMu4Zjx3ZK6/i0x3QnkoyHRYkXvad1KzW5QTN2opE+ftcwsI1h3MLMz8nYOvdL69gd+5WBmLbVHHC/L/rcOqeAsjlaW6Olh0Uai08u9LJdLx1gzXHrI7TXksGLG1YyXoQf3UALf/xQy/0Vnk7p8SM1wGTzKjr6twZPgww3S/zifHpDuwxy4Kivhbv14B0/fgytmD4e5enwByTZzwRCmdBPCjPK0luG1l1PfddMaUqeBD7zH2Nl+hiujcPo2s3F0d3RblXKa9QMx676ug1IPWmbxJC7yw1H7VhjMyQUC+SAUgNBhgWCLbCqMePHwOrTmCwsmdK8qNVGD4U6cOzEsTSPxY/ocx+YsfUwGnurVeeJOX7DZJxWOh1Btzlu0Tb+/Wgb97qdtm32uuBboBHZdsDsQm6RxFNS2oqllRou+0mUbalhuzKpff8NGZRO5UHpSiGTeYsODFq2eVFBmcrXfvMLCEDiLC2qFemK2RFBBhCuiejIwxz8SH7s7jMdI3i8M/6+B0Z7HU9v/F44Xa78njtw8xNKP+jQj7hapHIMvgrKH9AOzvfFWXSUt29k8Vih2LgkcWJLMT2hsgn42T167ZSSr3UlyicdmAviRt8sYUQ04ReOjgtTmifMeIF7msQzBeN6Vm3yWM8Q6/NdQvQZz06ZAucZKkP5W8F9blQGfzF2BjUbSD4bgjr5p57P0YpxahNC1/xDXFwHGkcaN0B7J1xCjYNPUTQ62x+h2Z6k1JrOEXYlnpn6CJWFuqFDUfFPHVOzNnk3s7G2hva5rzG43/jon3UfuzkJFS/k8A0gcui5xw6vWx1DoOiQ80iFtqPyAm0W/VmLp9AhrXZxLV7VD3XylKnCf1SqoXwlspbLZHIF09jq+JlUaGvPoPLv5E7nU4fhomAdh9Y5hhV21OZ4bId/0azr2IHFzY4xBg1VvMterDJ6eMztsJ2bM+5qih5tdczPO7WEE+yc4jjHg+aoroPPklpHCocTxbiFYuxEIU1T95hTdmuXnWrZF8ry1vyoXU1htyneZSkaTlRbjvm6qNH5uJk98LQjuqw9+77TmtEMI6mBNtBnj1HD7TMp1Cibx4tFLEyNkZDPULwkZUCfjFErxmS3MWbi1J+eCWBXDUbnAofpPEDyEK/2+El1dEig/lYuQM9Z6ziEORk6HNPDwc0g+hz9+7tffsDMMNENeKoRnTLg3oTkifPcRu16A/zGlsX0xpX6IY7BkWuGI5pw8dhqku7WUWQOUscOUsf/PUmtizrOsBqS8H8KaU7RsCODmlHesSWOVBWZvzj/ccWGoeCmC/B1UwqN5K7tazqu+IsTF7x+BK7bLMrZnO4eEcPFwNZVZgtGeQz1RpR2ITfO9ngGbNrI797vKE/Xx6MWA1wzWVHsx3B6araofFc2oi6WGsTBV8T2gDJGCDvCCU5B5aeAbxXNR4mocEG/YjZKRZss+YRAibp0RGPDeyd7tfnKWuZQbqCx75FGAR7B0nLXOL1p8sWVXibMmMbR4zBqVBkeoZj65bmXv0Etd6hKepeSS9dGi1hV/2atAVzOitpn+8vq1Zk2vEaV4SgJMFN48HBcXP5O802ig3cZTFjJC/KzQ/n9BpR8Js6XXXceR9uMHkp8rEZ0DJ6dFyHRdcQiRKabNEnUAaT8EB8W5e3sF7lYmBX53Le3YGnJtarqwLO22ZoVd+bm2fl5PY07yiPSSDKXPZG8zba7TUKT4tSm4+kV5zgFiMM/RY0hvPF7pNaMI2j0kIkz4hrz20cIpYuQrhVc/PNn2V5jAXrnIIKVpKZXqh2zKTOAnUlXoKymGZ7XZ3iHuJmOR+52TdKkxreOQ2rFE9xr3m7C9QyiLn9UEHqpeunaGDUdaTpWQ8ePbqj4p8lEbZJmHtFAQXygUdF2uexCTwx4OBohMLciChdWIMEl9CAxlW8g1QIiemt7j4ktiNTYR8YXFIcE+b9DkOEJ7jlYjwscdtaOWEewgq/wdgYs2t00lox6RORCSY+TSb9z+MIVe3Dy53PTMskVVxDCLOTK2yJJTRfaRAKGAlGMEbnAo8xmcXT4M2z/xoHqCGzYw7IpjHYzXYGkLqYSgTyr4ylEiCReYulrV/arvf4togmtQJ+rVCscqKrCBndU19sc8tgpoOPqx8cslnetkjuycHeI+LRHzy6qyK2OWtdpdOwuU5CKfx3hTze46L+NuauOkxU3JWL2OpTf7upMtL21L7Wti3bh0tjg0OaoqiOtwcWgdr7rBh49XdW7VFkXb3bSZrsMHteM38oe55cgGeOsbfp17x58RTdMQxGZUi7Xwtbi0uDES1Yl37rArweW7pd1Uuq2sUWNoQgAqahtV9G4UpXksHrE6pWqmQ7blLmfnAww4edo/beShVz+iY5DS0nZ4iZxm6vXdkq2zDxkWD6ZWWp0bV1xC7sziGVjfZ2WjmTelvupI+CNrst05TfvA5VLO+qsWXt4pj3Y93xthQ5FMtwsbRn5/+U/MheY8jN1sefxJczRq++L8vaLCDjRrZiBuhlTGgi0mt5YCI2ktFxy00u9JMDrtpsqZVOJNrX4Uqzan+bXPCOhnZ9tsqgB6ATi7ZJXZV4KB5xQ2Vl/Fi8kTHuoqFwECeFzrxg8/OWakogQvXzsta1OcYao5CTGfmSHiEiMYYK0w4Ad3WdkWzS12HGmVl1rx1YY8ZFO/um682W35BR+G2wkdjdb/MRb0cLpfeLjjoerPdGOPbmgTBfqHFih3qs9Cqzy/b0i+YHHWh+iyVfZTnPrUoHjCYZTCt6lRlDAr6WPilv9lldab4sn/HCAtaVwuZRB9+O1qn5v+9U46h2DbkptgTKIkT2Dy+d4bLIIA4O1cGybFlkxH13tnZP3lHhvSn4UD06DtJZC1I/4PzYk5uFCm9U17yLsx2/iw8hmdUCxUnVenXkF38df+AWjIj33oMT7lCo6sKRaFLcsBKGAZsWZvCYEZYgfNJMU9zluOG+u5DWu2RMP/J8snYQ40na8WE88VOuquTSPLZagPVlzWR5MQBS5zdv3dX/68AFDCYjMj1DpsijqhdCwIrtj0N/LGMWsorsqoZIQiQCjDM0QvP0QbzUgFCjFyTzhR+FsK0jM5kW/8Z437WbHx1769mn8f345+zSOzt5fjN98enVx9uE9LZzu74ttn/t043MpzkjBeQDbBVJ0B2yceQ54oRFkMWDjogAuduI0Lp/7idN3gbzxZZeF+Pg6rvRrz3lEX51KZG3Vp+EYk51I8u1bR+T1zJPFzBlvEid08o55vevV3ozuidl7CHAlRj8hFLXHQ4d/YxXtc2hx7Ti+OXywJ1K6815rwSe0eivfVydwBVpymPNUNKu10KBeazGETtVgFWXL4VyHehTelekXvqs/UrvPW9z9o4cd4tHCA0hYkSTedA2WHvyepDBWaswHlOoshUEU092keLqbSsjms6ZtlKANopuqwlfSXx0F3lHPslFyMe3SnQhY4hJHG0Jye+t649Kdf9gCgH+HaDZtAZQ1EQDeBq0jwOvRAYG9jSK2rHGgPe+11zPTwPsbNofa5eji+BJLYAw5vvwbcBl/DTaZP7El5TFi50UcFMYblgLwFLcYrz+P7evS+TxqXomuoFCoKuuC8ksV8YqzeaiLn7+E5vU2rVvpjflbY6BCv530rOvO7W3ZEa6ocIKae8+tW847LjJXlhKZxWDSiHY1V5V3Vsq2XFcuDNpjx454/AOfmHWNL8hf7AoIcL2BnSFuNY+zbJqh+4FIFBMAGiyZEXTWSW8nNNxxZ66bfzW8FEpvaZvqtoqq9QKXqddg8SR4PXoEOiHiyGK1jVvXNPZgAVE7Rg5pJfAOFLoGhYZCn+IrfWs0yPdDVWgoILRCYuVM+H58Je/ykq/n63/ThCL+udbnDtFBHfuYYj4Vo/bS5ilaitoARJdgH+L/rzk8f3FtodnQW18URPWltyMMM6qtWAfcmAqjz0eVHJ6CF4bZGpj1gf3lS99zIyz7VthvBDfWRo97rRePwNxmJIhE65JvPVwPst+SuSMSzz4fWLqALUFaltukBVFeC7bQ70ZuxON2yVHqAmv8ooE69cUmBiyRA06yHEpCWQcRGO/1mrRDXkQlKAq8xBFcaqeSIAtAnpV4aNtDMp7JqSLtSYktuGDl1rAd3NX1EPQPzdvBLhNUezmSg7cuoTe7ZhtrRcTV2a7d1Lk6xeOLOL2DbDTrGxPfujHy8cTPEPkizg7hBlSOLdoNkNmAOer5toyk/OjZiELiVTSL06wlEIYmJ4YPaEi1GO6wiHxL/1/2cS9AiIEx/NUPr687jEL9riJeH5qBYWO/2p/4LH+1l8foBfV2dAM0099hFu4y+TTe1s/omG/xtf6Nitr7m8Cz/2X5Gr/406zSKWnfTGR6t/zs2EgeMyR82y13UTsQkjQILFHFFnFea7e18xN4ox95EOIsnxWtS2343eXvi/onUO8JcbTrepuNjm5rgcK1m0ex9/w2XSKLMeXwrHGDPemETddQNr4DsUbP0301DnA8/FkcXbwz4xAImAIjZgriSDk62s41HxlmVXHGb5w2mdiIAN/ZyeOLwk8W2Y/8yMB2Qhk/vFkeZycpsTZJCVT2MUzNHZgk2pSGq7vnvGbT2T4yz17gZHUibo7Bd2PWL7k42kXxOy61wIzNWBye23Gm4ymFVR/c+YjU6z7JNj1jn+AeZQZK0W8rS3BlQI9hDIDmUv3iK5hYwSpHu5slLQUq7hrXqtBSgfiVMuLKbCzpvl8bl7PFzmZzLzAXi0B/ZYVfHbaaUUsgiNgEwvtU/J0toc3R58v+dbj5Pi0jCWUzNHi5R48kYPxNBIzbBBzvSkDBOYeL8hpi9faxaMZONOPHolk70ax7G+J45nm0rlMrHL4nLUHyY0zTKoLxFKe5Q+p/wzuVHWIvlGK3qJ9VZ+pg2YZZQWdtGyzwnbi0U1d1n7YdKGX2JFxSm252R+J83ZwwT4A8Bk1LQzLtH1+Im2ZrngQxrbkJ3FKHFbVfxT6XJVvSipG6C7rjWN6iCll+l5ZFLo4FfjuIMA7/w6vz8buz9+Pop7N3Y9dxeQj/VGNC3KQFBu7V3jlDsWvXCaYlnbICLdmnptC11qtl+1h3OrZVGh9Xe++LFgxv7LIs7tLEum2tWrIpLu8aizMhvo1w1PAlE7wqnu8Yxlsr8CQ9nKub2y5aB/qq8/ZbmPm2No4ba/HxL4uekE4vLkP2BagQ++AkRn1/K9neYB7iseMVg3F/MjjUiaCbgCVguMWe8tvWS8dUrOFHz2y7B7qN0JzduybS3fxwQcUO7iJUQ5yNtMEE0MbYCsSBK7Y2czpehxsdL+FzW0G+w3bcn789NCOxmKHXjtkNhKOmMKgv4bERsUPv4QtPxUIisJlGKjKv84sebiznlGGCUHT0ah1wsGYns46Uu122nwnTHuqsf6RL4qyVo/gd1oGbFaAjnJ9aEY4N/Y51OTQhDkylDekou/JO14c0ZTYG/v9qxG/WiMRRqRNp8eOfpRC3e2G/k0/0G/lDLj159HSF3u1MPTKCuI0ut/5W3nK3YfjkyN+3qHKKPln72awRiagclh3fUHO5P7gWR9WHmMUW+WF/0Eq6un86gv+da/6HzDVq76O4nLiC0b2IKSDB0+LNU5yNQO6ubtS6s2NE3HYQGp0hkxMiDPdz16G1Q9PelcIzy0Z9fScbrhc3uzXxW3tT5mhgnwSh9miaJKMEoBytR2qnW2DtzmylPZBw9GQP8wUQmUq2oe86QvZ58PhoPa2DPGb1GJfELrbWQ8tzzeJ8b8si3A6LHGobd7OgYUmqdjOKvAGJxFTeQbOLfOrT6O2y5jNn+4q7U3wt77AvlrgPmy3xh/CYMRr78GDVFDpuasZKNjng+N26FfnSuJgSRoidUWbknPKbhjobKQuEjrtyLdpUUXtOajvvG2el9pqIfp+QU+p1h2kHSidlccv45lO9a5yl2szVzhYFFMe/FfVOwjkNDqWLFqYpxXecSkzfpPRPMMJEKv6T1e5/YwlXF/Bdmwn3RsCtA7a5LpXnfOxWewPluqbaFS1zDy43q///jjOcXW7ZWjtQow3bca7GJt1P2IKGQB3dJdR33bNomKUsSzQqfP2+SZGDg5OQdeFir7X3Otk2mEUxWxiIABzGRMJOC7SbhrM1pPWudo5uStz8HdWHxQSMEqGUPsJPkSDR03SBBG+pA/XBoRGMOn8zBm3rjK5aedZK5wSZ8WsMH80ZjrbFF/HawRVeUavmTarRTIBrUNtNV6jtWWad1zesTqcRnfASaWe6aLZUyRCfe7lluSb9EGXo5+GRM3KN5QIR6ksq7uLhFGw5DNaolX9aIm3uiLUQcPA5K/hxOSFdKCAgf0AbL/BwudsFFfKtPzoEFn1HQRdrR5fMDrV8kr7cADpEv2RASV6UidjyTQAWkftfRpqDKxMtVQ0j7bzTjuCLOgB1ZNMivaKwZEuRTNq50Rd8ki6/1zimDlkhSOd88S+pOdfed96R7ThNiwwzw6NZPiLWhxTdInsFt9e0ghFiR0d4wZD7cbn+MS0ZD17xI8VGStQbwcRbG+PKS2RRa/xvMXX4lXt2npiUOYCzpM2HejCmN1L14RI4uB9FOaIDivRDz3bOHUO6RnSAmLyZFHAso1vwLPlTFletKZzn4tAgovsw6ZcfT6csY2XMCZouV4iR7+VER9XDg8VoN1XbbW3/ydJFWouhP+FLativ/DU0Vnu5DRX6xFWc4zmtCMgd5z6eCzefs5J7v1uR8AAl7jKlTUlQvYRUXVaNLuXPazezwlkqLKVACLT815GlDzYLmSXiiCvjSDu7vzttgW6XVZIkSY7kAWu2rSTp4VeAAkESIpzQCjilXPFtZuoLCo32ZRfyaIMUFQYaKcTdygu0bVoaINwSpeGtGx9dlGwJDSjzsbXVSUsQkm+3xAgUrh0XyzcT9KQ182+0t3eyufm+xU2WtzgGXKQHpjMvinBtPYr4uYhRRLehROpoRL51UZyzTJltWAAvNZ8UFQxjVATsSyr3F4a8PIcVK0H8VXgfV+crUEtVNVuZN8jhFjLEobZU8Ntfr/Y+vjo/P5Vhi5/Gry5++TSOLsbnF+eKs1vXlJy4tKLRq/c/Rud/fX/xdnxx9jo6//nDv5vrWSaCi798ePvhY/Tz+NX76OO7X86jNkYFbW3Wl4j0dMC08rgkxhlKQ30DL1AkvvfwcBiYWjxcuVriLe+YvunFk+KOhaoGidLejCkP9gu8t8ODNyhqMu4Z0EZNUi8eLoCsOB1gr1K2ScJx4+a3vbsBHos3PD5B63Xv4u3Zpx+jj68+Xfw1ev8BWDWWbQ8XCe4cnZ4Mhux40j8eJsnxiwE7TJ6/OBnOjuOTF4PZYMYG7GX/8GTKjg6ng+OjweGwP3vJkj7rn8yS4SF7gRtT+f281QGNigOZ9Sk2qK7jRYYVnbDhdMjil5PDATsaJJOXs8Ex6w9P4iE7GR4P+xPWnyXskE1fnLzox4fPk5cng8lhPDyZnRyzSX+AFSXz5EClqh5YabRiK+zksD+bxSfTwXP2/MVk+HyYnCQvD+HX5PA4eTGcHcZH0ySeTdjsMGFHx0N4hmZNhvH0JTvpT09a9ZgpsaKWeDibPJ8ewsCANvVP4uQwfvkimT2P48Mhez54/hwqmgwGQ6hgOoGWPE+S5y9P4pOXyWDAJmy4ZT/vyXE/GRz1Tw7Z7GhyyJ6zly/6Ry+mL17Mjl8cxS+fHw1esv7L4WA6m82AhoTFxycvjvpHRwy6bHo02Ht4+H91L4Je"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def target(root, name):
    path = root / name
    if not path.resolve().is_relative_to(root) or path.is_symlink():
        raise RuntimeError("Unsafe path: " + name)
    return path


def atomic_write(path, raw, mode=0o644):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".mean_hog_", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_payload():
    raw = zlib.decompress(base64.b64decode(PAYLOAD, validate=True))
    if sha(raw) != PAYLOAD_SHA:
        raise RuntimeError("Installer payload checksum failed")
    bundle = json.loads(raw)
    outputs = {name: text.encode("utf-8") for name, text in bundle["files"].items()}
    if set(outputs) != ALLOWED or set(bundle["v1_sha256"]) != ALLOWED - {STACK_CFG}:
        raise RuntimeError("Unexpected payload target set")
    for name, content in outputs.items():
        if name.endswith(".py"):
            compile(content.decode("utf-8"), name, "exec")
    return outputs, bundle["v1_sha256"]


def preflight(root, outputs, v1):
    before = {}
    for name in outputs:
        path = target(root, name)
        if path.exists() and not path.is_file():
            raise RuntimeError("Target is not a file: " + name)
        before[name] = path.read_bytes() if path.exists() else None
    if all(before[name] == content for name, content in outputs.items()):
        return "already-installed", before
    if before[H1] is None:
        raise RuntimeError("H1 source not found")
    current_sha = sha(before[H1])
    if current_sha == BASE_H1_SHA:
        for name, raw in before.items():
            if name != H1 and raw is not None:
                raise RuntimeError("Original-H1 mode refuses existing HOG target: " + name)
        mode = "original-mean-capable-source"
    elif current_sha == v1[H1]:
        for name, expected in v1.items():
            if before[name] is None or sha(before[name]) != expected:
                raise RuntimeError("First-HOG-patch file differs; no files changed: " + name)
        if before[STACK_CFG] is not None:
            raise RuntimeError("Refusing to overwrite an existing stack config")
        mode = "upgrade-first-HOG-patch"
    else:
        raise RuntimeError(
            "Unknown H1 source SHA256; no files changed: " + current_sha)
    return mode, before


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    outputs, v1 = load_payload()
    mode, before = preflight(root, outputs, v1)
    print("MODE=" + mode)
    print("PASS: embedded payload, accepted source files, syntax and collisions")
    if mode == "already-installed":
        print("PASS: TWOHOP_MEAN_HOG_OFFLINE_INSTALLED (already installed; no writes)")
        return
    changes = {name: raw for name, raw in outputs.items() if raw != before[name]}
    for name in sorted(outputs):
        action = "UNCHANGED" if name not in changes else (
            "MODIFY" if before[name] is not None else "ADD")
        print(action + " " + name)
    print("New config: model=beta_h1_twohop_mean_hog_rw")
    print("two_hop.enabled=true; normalization=mean; hog_rw.enabled=true; rw_depth=4")
    if not args.apply:
        print("Dry run only. Add --apply to install.")
        return
    parent = target(root, "interface_patch_backups")
    parent.mkdir(parents=True, exist_ok=True)
    backup = Path(tempfile.mkdtemp(
        prefix="h1_mean_hog_" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_",
        dir=parent))
    modes = {}
    for name in changes:
        path = target(root, name)
        if before[name] is not None:
            if path.read_bytes() != before[name]:
                raise RuntimeError("File changed after preflight: " + name)
            saved = backup / name
            saved.parent.mkdir(parents=True, exist_ok=True)
            saved.write_bytes(before[name])
            modes[name] = path.stat().st_mode & 0o777
    # Ensure the exact pre-install H1 is available for disabled-feature regression.
    baseline_file = backup / H1
    if not baseline_file.exists():
        baseline_file.parent.mkdir(parents=True, exist_ok=True)
        baseline_file.write_bytes(before[H1])
    written = []
    try:
        order = sorted(set(changes) - {H1}) + ([H1] if H1 in changes else [])
        for name in order:
            path = target(root, name)
            current = path.read_bytes() if path.exists() else None
            if current != before[name]:
                raise RuntimeError("Concurrent file change: " + name)
            atomic_write(path, changes[name], modes.get(name, 0o644))
            written.append(name)
        for name, expected in outputs.items():
            if target(root, name).read_bytes() != expected:
                raise RuntimeError("Post-install verification failed: " + name)
        manifest = {
            "mode": mode, "backup": str(backup), "baseline_file": str(baseline_file),
            "new_config": "beta_h1_twohop_mean_hog_rw",
            "modified": sorted(changes),
            "before": {name: sha(raw) if raw is not None else None for name, raw in before.items()},
            "after": {name: sha(raw) for name, raw in outputs.items()},
        }
        (backup / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        state = target(root, STATE)
        state.mkdir(parents=True, exist_ok=True)
        atomic_write(state / "backup_path.txt", (str(backup) + "\n").encode())
        atomic_write(state / "baseline_file.txt", (str(baseline_file) + "\n").encode())
    except BaseException:
        for name in reversed(written):
            path = target(root, name)
            if path.exists() and path.read_bytes() == changes[name]:
                if before[name] is None:
                    path.unlink()
                else:
                    atomic_write(path, before[name], modes[name])
        raise
    print("BACKUP=" + str(backup))
    print("BASELINE_FILE=" + str(baseline_file))
    print("PASS: TWOHOP_MEAN_HOG_OFFLINE_INSTALLED")
    print("No formal training/sampling, data/cache changes or network requests.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("STOP: " + str(error), file=sys.stderr)
        sys.exit(1)

