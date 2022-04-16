"""
A script for creating a plot of a model's loss curves. The raw data needs to be
pasted in manually.
"""

from os import path
import matplotlib.pyplot as plt

output_path = 'outputs/losses_skipdecode.png'
num_iterations = 50000
interval = 1000

iterations = list(range(interval, num_iterations + interval, interval))
gen_losses = [2.37159546382213, 2.868055909084156, 3.3169256161209195, 3.8102839904464783, 4.083530010595843, 4.374707531958818, 4.689916086278855, 5.1210505489632485, 5.56638439399004, 5.739707717701792, 5.540943785905838, 5.5443742139935495, 5.4242067390866575, 5.4277886128276585, 5.621797967851162, 5.586387036953122, 5.542782301768661, 5.623747801735997, 5.6773548617847265, 5.529059382900596, 6.180996619537472, 5.489048743609339, 5.3960974473953245, 5.5058905618339775, 5.584908930908888, 5.6057321792077275, 5.561913089841604, 5.781813509471714, 5.790913476735353, 5.808147210597992, 5.555723754353822, 5.915932406110689, 5.697399411156773, 6.224352464139462, 5.82181827763468, 5.951231693297625, 6.2436938397772614, 6.121177977606655, 5.880357255958021, 6.019801439478994, 5.924496628567576, 6.168394108030945, 6.12925958615914, 6.366405184932053, 6.156207211852074, 6.484298097968101, 6.276147607401013, 6.224191000401974, 6.20815112157911, 6.24299076656159]
disc_losses = [0.6600233054792043, 0.34639124221756357, 0.22720127323637826, 0.15507000366472448, 0.13673935526271544, 0.13635928407879191, 0.12755716085584298, 0.13238659342702339, 0.14296927614986998, 0.17808428480242947, 0.17528971174914112, 0.18743290280802102, 0.1847826884310939, 0.1757875422563411, 0.17248653895984303, 0.179771073820948, 0.18048924703412286, 0.16665095142545397, 0.1594565645763887, 0.13727913542545594, 0.1453402533666449, 0.16424614142506108, 0.1279481972958256, 0.1475687136623492, 0.1410423647311086, 0.13834850343486368, 0.12196173002654541, 0.12384169733064072, 0.14157487485676576, 0.1145171155883054, 0.1241907245056791, 0.10506548571728194, 0.10160618337756205, 0.10629433530597816, 0.08962243948444666, 0.11448525415517724, 0.09425660197245907, 0.09897890687017935, 0.11443760216599548, 0.09254841286055217, 0.0945449370062779, 0.09812727364719831, 0.09204184845329291, 0.08615197649701077, 0.09292185680223565, 0.08797294501229072, 0.0914810908769141, 0.08895265739900969, 0.07471466737144146, 0.07738279180326026]
recon_losses = [0.3822403694987297, 0.31723701363801954, 0.28870812329649925, 0.26748799833655357, 0.2585953095853329, 0.24843679656088352, 0.23619291043281554, 0.2277828796505928, 0.22986686623096467, 0.2254442735463381, 0.22230030591785907, 0.2245340961664915, 0.21814753444492818, 0.21648146642744542, 0.21378481233119964, 0.211496033847332, 0.20890579986572266, 0.20366672165691851, 0.20451605071127416, 0.19878139005601406, 0.19844229586422443, 0.19725102286040783, 0.1936472097784281, 0.19206482180953025, 0.19204924750328065, 0.18956122170388698, 0.18847558185458183, 0.18572783245146274, 0.1861315475255251, 0.1833372358083725, 0.18309373919665814, 0.1806995974481106, 0.17963291117548943, 0.1803018360286951, 0.17612652230262757, 0.17637226301431655, 0.1743457129597664, 0.1745091659873724, 0.17203972665965558, 0.1716908378303051, 0.17157007364928722, 0.1703660314232111, 0.16942141018807888, 0.1682369754910469, 0.16795882569253445, 0.16689433626830577, 0.16568922270834446, 0.1640736695677042, 0.16531087101995945, 0.16300042364001274]

fig, ax = plt.subplots()
ax.set(xlabel='Iteration', ylabel='Loss', ylim=(0, 7))

ax.plot(iterations, gen_losses, color='b', label='Generator')
ax.plot(iterations, disc_losses, color='r', label='Discriminator')
if recon_losses is not None:
    ax.plot(iterations, recon_losses, color='g', label='Reconstruction')

ax.legend()

fig.tight_layout()
fig.savefig(output_path)